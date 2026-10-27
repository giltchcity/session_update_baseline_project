// Read-only full-current-surface proximity to fixed, method-independent probes.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include <khronos/spatio_temporal_map/spatio_temporal_map.h>
#include <khronos/utils/khronos_attribute_utils.h>
using J=nlohmann::json;
using V=Eigen::Vector3d;
struct Key { int x,y,z; bool operator==(const Key&o)const{return x==o.x&&y==o.y&&z==o.z;} };
struct Hash {size_t operator()(const Key&k)const{return size_t(k.x)*73856093u ^ size_t(k.y)*19349663u ^ size_t(k.z)*83492791u;}};
constexpr double cell=.20, radius=.10;
Key key(const V&p){return {int(std::floor(p.x()/cell)),int(std::floor(p.y()/cell)),int(std::floor(p.z()/cell))};}
V closest(const V&p,const V&a,const V&b,const V&c){
  // Real-Time Collision Detection, closest point on triangle (including edges).
  V ab=b-a,ac=c-a,ap=p-a;double d1=ab.dot(ap),d2=ac.dot(ap);
  if(d1<=0&&d2<=0)return a;V bp=p-b;double d3=ab.dot(bp),d4=ac.dot(bp);
  if(d3>=0&&d4<=d3)return b;double vc=d1*d4-d3*d2;
  if(vc<=0&&d1>=0&&d3<=0)return a+(d1/(d1-d3))*ab;
  V cp=p-c;double d5=ab.dot(cp),d6=ac.dot(cp);if(d6>=0&&d5<=d6)return c;
  double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return a+(d2/(d2-d6))*ac;
  double va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);
  double denom=va+vb+vc;if(std::abs(denom)<1e-24)return a;
  return a+(vb/denom)*ab+(vc/denom)*ac;
}
struct Hit {double d2=radius*radius+1e-12;V p=V::Zero();uint64_t owner=0;int layer=-1;};
int main(int argc,char**argv){try{
  if(argc==2&&std::string(argv[1])=="--self-test"){
    const V a(0,0,0),b(2,0,0),c(0,2,0);
    const std::vector<std::pair<V,V>> cases={{{.5,.5,1},{.5,.5,0}},{{3,0,0},{2,0,0}},{{-1,-1,0},{0,0,0}},{{1.5,1.5,0},{1,1,0}},{{.3,.4,0},{.3,.4,0}}};
    for(const auto&t:cases)if((closest(t.first,a,b,c)-t.second).norm()>1e-10)throw std::runtime_error("Triangle closest-point self-test failed");
    std::cout<<"5 exact triangle cases passed (including a large face with no nearby vertex).\n";return 0;
  }
  if(argc!=4)throw std::runtime_error("Usage: cleanup_mesh MAP PROBES_JSON OUTPUT_JSONL");
  J groups;std::ifstream(argv[2])>>groups;std::vector<V> probes;std::vector<std::pair<std::string,std::pair<size_t,size_t>>> offsets;
  for(const auto&g:groups){size_t start=probes.size();for(const auto&p:g["points"])probes.emplace_back(p[0].get<double>(),p[1].get<double>(),p[2].get<double>());offsets.push_back({g["event_id"],{start,probes.size()}});}
  std::unordered_map<Key,std::vector<size_t>,Hash> grid;for(size_t i=0;i<probes.size();++i)grid[key(probes[i])].push_back(i);
  std::cerr<<"Loading "<<argv[1]<<" probes="<<probes.size()<<std::endl;
  auto map=khronos::SpatioTemporalMap::load(argv[1]);if(!map)throw std::runtime_error("Failed loading map");std::ofstream out(argv[3]);
  for(size_t ix=0;ix<map->numTimeSteps();++ix){
    const auto stamp=map->stamps()[ix];auto dsg=map->getDsgPtr(stamp);std::vector<Hit> hits(probes.size()),history(probes.size());size_t faces=0,invalid=0,vertices=0;
    auto mesh=[&](const spark_dsg::Mesh&m,int layer,uint64_t owner,const khronos::KhronosObjectAttributes*a=nullptr){
      vertices+=m.numVertices();std::vector<V> world;world.reserve(m.numVertices());for(size_t i=0;i<m.numVertices();++i)world.push_back(a?a->bounding_box.pointToWorldFrame(m.pos(i)).cast<double>():m.pos(i).cast<double>());
      auto& target=layer==2?history:hits;
      for(size_t f=0;f<m.numFaces();++f){const auto& t=m.face(f);if(t[0]>=world.size()||t[1]>=world.size()||t[2]>=world.size()){++invalid;continue;}
        const V&a=world[t[0]],&b=world[t[1]],&c=world[t[2]];if(!a.allFinite()||!b.allFinite()||!c.allFinite()){++invalid;continue;}++faces;
        Key lo=key(a.cwiseMin(b).cwiseMin(c)-V::Constant(radius));Key hi=key(a.cwiseMax(b).cwiseMax(c)+V::Constant(radius));
        if(int64_t(hi.x-lo.x+1)*(hi.y-lo.y+1)*(hi.z-lo.z+1)>1000000)throw std::runtime_error("Unexpected giant triangle");
        for(int x=lo.x;x<=hi.x;++x)for(int y=lo.y;y<=hi.y;++y)for(int z=lo.z;z<=hi.z;++z){auto it=grid.find({x,y,z});if(it==grid.end())continue;
          for(size_t i:it->second){V q=closest(probes[i],a,b,c);double d2=(probes[i]-q).squaredNorm();if(d2<target[i].d2){target[i]={d2,q,owner,layer};}}
        }
      }
    };
    if(dsg->hasMesh()&&dsg->mesh())mesh(*dsg->mesh(),0,0);
    if(dsg->hasLayer(khronos::DsgLayers::OBJECTS))for(const auto&[id,node]:dsg->getLayer(khronos::DsgLayers::OBJECTS).nodes()){
      const auto*a=node->tryAttributes<khronos::KhronosObjectAttributes>();if(a)mesh(a->mesh,khronos::isPresent(*a,stamp)?1:2,id,a);
    }
    J row={{"index",ix},{"stamp_ns",stamp},{"faces",faces},{"vertices",vertices},{"invalid_faces",invalid},{"events",J::object()},{"history_events",J::object()}};
    for(const auto&o:offsets){for(int h=0;h<2;++h){J values=J::array();const auto&src=h?history:hits;for(size_t i=o.second.first;i<o.second.second;++i){const auto&r=src[i];if(r.layer<0)values.push_back(nullptr);else values.push_back({std::sqrt(r.d2),r.p.x(),r.p.y(),r.p.z(),r.layer,std::to_string(r.owner)});}row[h?"history_events":"events"][o.first]=std::move(values);}}
    out<<row.dump()<<'\n';out.flush();if(!out)throw std::runtime_error("Output failed");std::cerr<<ix+1<<"/"<<map->numTimeSteps()<<" faces="<<faces<<std::endl;
  }
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
