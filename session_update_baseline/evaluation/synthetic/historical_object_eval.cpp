#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
#include <khronos_eval/object_evaluator.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
using J=nlohmann::json;namespace fs=std::filesystem;
int main(int argc,char**argv){
 if(argc!=4)return 2;hydra::PipelineConfig config;hydra::GlobalInfo::init(config);
 auto map=khronos::SpatioTemporalMap::load(argv[1]);auto man=J::parse(std::ifstream(argv[2]));fs::path out(argv[3]);fs::create_directories(out);
 khronos::ObjectEvaluator::Config cfg;cfg.association_method=khronos::ObjectEvaluator::Config::AssociationMethod::SURFACE;
 cfg.max_matching_distance=0.5;cfg.surface_subsampling_rate=1;cfg.match_require_semantics=true;cfg.compensate_missegmentation=true;
 khronos::ObjectEvaluator ev(cfg);ev.setOutputFile((out/"object.csv").string(),(out/"vis").string());
 std::vector<khronos::Points> local;
 for(const auto&o:man["objects"]){std::ifstream f(o["points"].get<std::string>(),std::ios::binary);khronos::Points p;float v[3];while(f.read((char*)v,12))p.emplace_back(v[0],v[1],v[2]);local.push_back(p);}
 for(size_t ix=0;ix<man["queries"].size();++ix){
  const auto&q=man["queries"][ix];auto t=q["query_stamp_ns"].get<uint64_t>();
  auto raw=q["snapshot_stamp_ns"].is_null()?std::make_shared<khronos::DynamicSceneGraph>():map->getDsgPtr(q["snapshot_stamp_ns"].get<uint64_t>());
  auto pred=std::make_shared<khronos::DynamicSceneGraph>(),gt=std::make_shared<khronos::DynamicSceneGraph>();
  std::set<std::string> ignore;for(const auto&id:q["ignore_prediction_nodes"])ignore.insert(id.get<std::string>());
  for(const auto&[id,n]:raw->getLayer(khronos::DsgLayers::OBJECTS).nodes()){
   if(!ignore.count(std::to_string(id)))pred->emplaceNode(khronos::DsgLayers::OBJECTS,id,n->attributes().clone());
  }
  for(const auto&u:man["units"]){
   khronos::Points pts;bool present=false;
   for(const auto&j0:u["members"]){size_t j=j0.get<size_t>();bool visible=q["visible"][j].get<bool>();present|=visible;
    Eigen::Matrix4f T;for(int r=0;r<4;r++)for(int c=0;c<4;c++)T(r,c)=q["transforms"][j][r][c].get<float>();
    for(const auto&p:local[j])pts.push_back(T.block<3,3>(0,0)*p+T.block<3,1>(0,3));
   }
   auto a=std::make_unique<khronos::KhronosObjectAttributes>();a->name=u["name"].get<std::string>();a->semantic_label=u["semantic"].get<uint32_t>();
   a->first_observed_ns={0};a->last_observed_ns={present?std::numeric_limits<uint64_t>::max():0};
   Eigen::Vector3f center=Eigen::Vector3f::Zero();for(const auto&p:pts)center+=p;center/=pts.size();
   a->bounding_box=khronos::BoundingBox(pts);a->bounding_box.world_P_center=center;a->position=center.cast<double>();
   a->mesh.resizeVertices(pts.size());for(size_t k=0;k<pts.size();++k)a->mesh.setPos(k,pts[k]-center);
   gt->emplaceNode(khronos::DsgLayers::OBJECTS,khronos::NodeSymbol('O',u["id"].get<uint64_t>()),std::move(a));
  }
  ev.setGroundTruthDSG(gt);ev.setEvalDSG(pred);if(!ev.evaluate(std::to_string(ix),t))return 4;
  std::cout<<"GROUP_OBJECT "<<ix+1<<"/"<<man["queries"].size()<<std::endl;
 }
}
