#include <cstdlib>
#include <iostream>
#include <limits>
#include <hydra/input/camera.h>
#include <hydra/input/sensor_extrinsics.h>
#include <khronos/active_window/object_extraction/mesh_object_extractor.h>
#include <khronos/utils/geometry_utils.h>
using namespace khronos;
namespace {
void require(bool value, const char* message) {
  if (!value) { std::cerr << "FAILED: " << message << '\n'; std::exit(1); }
}
FrameData::Ptr makeFrame(TimeStamp t, float object_x, float camera_x = 0,
                        bool occluder = false, bool missing_depth = false) {
  hydra::Camera::Config c; c.width=96; c.height=72; c.fx=c.fy=80;
  c.cx=48; c.cy=36; c.min_range=.1; c.max_range=10;
  c.extrinsics=hydra::ParamSensorExtrinsics::Config();
  auto camera=std::make_shared<hydra::Camera>(c,"consistency");
  hydra::InputData input(camera); input.timestamp_ns=t;
  input.world_T_body=Eigen::Isometry3d::Identity(); input.world_T_body.translation().x()=camera_x;
  input.depth_image=cv::Mat(72,96,CV_32FC1,cv::Scalar(3));
  input.color_image=cv::Mat(72,96,CV_8UC3,cv::Scalar(128,128,128));
  input.label_image=cv::Mat(72,96,CV_32SC1,cv::Scalar(0));
  cv::Mat labels=cv::Mat::zeros(72,96,CV_32SC1);
  MeasurementCluster cluster; cluster.id=7;
  for (int v=0;v<72;++v) for (int u=0;u<96;++u) {
    float x=camera_x+(u-48)/80.f, y=(v-36)/80.f;
    if (std::abs(x-object_x)<.2f && std::abs(y)<.25f) {
      input.depth_image.at<float>(v,u)=1;
      labels.at<int>(v,u)=7;
      cluster.pixels.emplace_back(u,v);
    }
    if (occluder && u<48) {
      input.depth_image.at<float>(v,u)=.5;
      labels.at<int>(v,u)=0;
    }
  }
  if(occluder) cluster.pixels.erase(std::remove_if(cluster.pixels.begin(),cluster.pixels.end(),
      [&](const auto& p){return labels.at<int>(p.v,p.u)!=7;}),cluster.pixels.end());
  require(camera->finalizeRepresentations(input,true),"valid camera frame");
  if (missing_depth) input.range_image.setTo(std::numeric_limits<float>::quiet_NaN());
  auto f=std::make_shared<FrameData>(input); f->object_image=labels;
  f->semantic_clusters.push_back(cluster); return f;
}
std::vector<std::pair<FrameData::Ptr,int>> select(const std::vector<FrameData::Ptr>& fs) {
  FrameDataBuffer::Config bc; bc.max_buffer_size=100; bc.store_every_n_frames=1;
  FrameDataBuffer buffer(bc); Track track{}; track.physical_instance_id=7;
  // No dynamic flag: the detector may have missed motion, or D1 rejected it.
  for(const auto& f:fs) {buffer.storeData(f); track.observations.emplace_back(f->input.timestamp_ns,7,-1);}
  MeshObjectExtractor::Config config; config.mesh_integrator.integrator_threads=1;
  MeshObjectExtractor extractor(config);
  return extractor.selectStaticFrames(track,buffer);
}
}
int main() {
  hydra::PipelineConfig pc; pc.default_num_threads=1; hydra::GlobalInfo::init(pc);
  auto moved=select({makeFrame(1,0),makeFrame(2,.1),makeFrame(3,.2),makeFrame(4,.2)});
  require(moved.size()==2 && moved.front().first->input.timestamp_ns==3,
          "unlabelled sub-metre motion cannot contaminate the current static mesh");
  auto still=select({makeFrame(1,0,-.1),makeFrame(2,0,0),makeFrame(3,0,.1)});
  require(still.size()==3,"camera motion preserves multi-view stationary geometry");
  auto hidden=select({makeFrame(1,0),makeFrame(2,0,0,true)});
  require(hidden.size()==2,"foreground occlusion is not object motion");
  auto outside=select({makeFrame(1,0,0),makeFrame(2,2,2)});
  require(outside.size()==2,"nonoverlapping views are unknown, not displacement evidence");
  auto invalid=select({makeFrame(1,0,0,false,true),makeFrame(2,.2,0,false,true)});
  require(invalid.size()==2,"missing depth does not split a state");
  auto returned=select({makeFrame(1,0),makeFrame(2,.2),makeFrame(3,0),makeFrame(4,0)});
  require(returned.size()==2,"a return to an old pose does not bridge an intervening motion state");
  std::cout<<"static_surface_consistency_tests_passed\n";
}
