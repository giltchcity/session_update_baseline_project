#include <fstream>
#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
using J=nlohmann::json;
int main(int argc,char**argv){
  if(argc!=3)return 2;
  auto map=khronos::SpatioTemporalMap::load(argv[1]);
  if(!map||!map->numTimeSteps())return 3;
  std::filesystem::create_directories(argv[2]);
  std::ofstream out(std::string(argv[2])+"/snapshots.jsonl");
  for(size_t ix=0;ix<map->numTimeSteps();++ix){
    auto dsg=map->rawDsg(ix);auto stamp=map->stamps()[ix];
    J snap={{"index",ix},{"stamp_ns",stamp},{"objects",J::array()}};
    if(dsg->hasLayer(khronos::DsgLayers::OBJECTS))
    for(const auto&[id,node]:dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()){
      auto a=node->tryAttributes<khronos::KhronosObjectAttributes>();if(!a)continue;
      J row={{"id",std::to_string(id)},{"semantic",a->semantic_label},
        {"first_observed_ns",a->first_observed_ns},{"last_observed_ns",a->last_observed_ns},
        {"present",khronos::isPresent(*a,stamp)},{"mesh_vertices",a->mesh.numVertices()},
        {"observation_first_ns",khronos::observationFirstStamp(*a)},
        {"observation_last_ns",khronos::observationLastStamp(*a)},
        {"details",a->details},{"centroid",nullptr},{"surface",J::array()},
        {"trajectory_timestamps",a->trajectory_timestamps},{"trajectory_positions",J::array()}};
      for(const auto&p:a->trajectory_positions)row["trajectory_positions"].push_back({p.x(),p.y(),p.z()});
      Eigen::Vector3d sum=Eigen::Vector3d::Zero(),lo=Eigen::Vector3d::Constant(1e30),hi=Eigen::Vector3d::Constant(-1e30);
      size_t stride=std::max<size_t>(1,(a->mesh.numVertices()+1499)/1500),j=0;
      for(const auto&p:a->mesh.points){
        Eigen::Vector3d w=a->bounding_box.pointToWorldFrame(p).cast<double>();sum+=w;lo=lo.cwiseMin(w);hi=hi.cwiseMax(w);
        if(j++%stride==0)row["surface"].push_back({w.x(),w.y(),w.z()});
      }
      if(a->mesh.numVertices()){
        sum/=a->mesh.numVertices();row["centroid"]={sum.x(),sum.y(),sum.z()};
        row["bbox_min"]={lo.x(),lo.y(),lo.z()};row["bbox_max"]={hi.x(),hi.y(),hi.z()};
      }
      snap["objects"].push_back(std::move(row));
    }
    out<<snap.dump()<<"\n";out.flush();if(!out)return 4;
    std::cout<<ix+1<<"/"<<map->numTimeSteps()<<" "<<stamp<<std::endl;
  }
  std::ofstream meta(std::string(argv[2])+"/COMPLETE.json");
  meta<<J({{"map",argv[1]},{"snapshots",map->numTimeSteps()},{"provenance","mapper_4dmap_raw_stored_snapshots"}}).dump(2);
}
