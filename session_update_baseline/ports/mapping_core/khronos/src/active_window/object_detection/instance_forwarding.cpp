/** -----------------------------------------------------------------------------
 * Copyright (c) 2024 Massachusetts Institute of Technology.
 * All Rights Reserved.
 *
 * AUTHORS:      Lukas Schmid <lschmid@mit.edu>, Marcus Abate <mabate@mit.edu>,
 *               Yun Chang <yunchang@mit.edu>, Luca Carlone <lcarlone@mit.edu>
 * AFFILIATION:  MIT SPARK Lab, Massachusetts Institute of Technology
 * YEAR:         2024
 * SOURCE:       https://github.com/MIT-SPARK/Khronos
 * LICENSE:      BSD 3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * -------------------------------------------------------------------------- */

#include "khronos/active_window/object_detection/instance_forwarding.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <opencv2/imgproc.hpp>

#include <string>
#include <vector>

#include "khronos/utils/geometry_utils.h"

namespace khronos {

namespace {

bool isValidObjectMeasurementPixel(const InputData& input,
                                   int u,
                                   int v,
                                   float detector_max_range) {
  if (input.range_image.empty() || input.vertex_map.empty() || u < 0 || v < 0 ||
      u >= input.range_image.cols || v >= input.range_image.rows ||
      u >= input.vertex_map.cols || v >= input.vertex_map.rows) {
    return false;
  }

  const float range = input.range_image.at<InputData::RangeType>(v, u);
  if (!std::isfinite(range) || range <= 0.0f || !input.inRange(range) ||
      (detector_max_range > 0.0f && range > detector_max_range)) {
    return false;
  }

  const auto& vertex = input.vertex_map.at<InputData::VertexType>(v, u);
  return std::isfinite(vertex[0]) && std::isfinite(vertex[1]) &&
         std::isfinite(vertex[2]);
}

}  // namespace

void declare_config(InstanceForwarding::Config& config) {
  using namespace config;
  name("InstanceForwarding");
  field(config.verbosity, "verbosity");
  field(config.max_range, "max_range", "m");
  field(config.min_cluster_size, "min_cluster_size");
  field(config.max_cluster_size, "max_cluster_size");
  field(config.min_object_volume, "min_object_volume", "m");
  field(config.max_object_volume, "max_object_volume", "m");
  field(config.max_background_score, "max_background_score");
  config.background.setOptional();
  field(config.background, "background");
  config.metric.setOptional();
  field(config.metric, "metric");
}

InstanceForwarding::InstanceForwarding(const Config& config)
    : config(config::checkValid(config)),
      filter_by_volume_(config.min_object_volume > 0.0 || config.max_object_volume > 0.0),
      background_(config.background.create()),
      metric_(config.metric.create()) {
  if (background_) {
    CHECK(metric_) << "Specify the metric if background is set.";
  }
}

void InstanceForwarding::processInput(const VolumetricMap& /* map */, FrameData& data) {
  processing_stamp_ = data.input.timestamp_ns;
  Timer timer("object_detection/all", processing_stamp_);

  // README principle 5: the semantic class never decides a motion cluster; whether an identified
  // object moves is the recursion (5r) of the tracker.
  extractSemanticClusters(data);
}

void InstanceForwarding::extractSemanticClusters(FrameData& data) {
  // Cluster by physical instance id when one was supplied, and keep the semantic
  // label separate so a cluster carries both "which object it is" and "what it
  // is". Without an instance channel this falls back to the upstream behaviour of
  // clustering the label image directly.
  const bool have_instances = !data.instance_image.empty();
  const cv::Mat& cluster_image = have_instances ? data.instance_image : data.input.label_image;
  data.object_image = cv::Mat::zeros(cluster_image.size(), CV_32SC1);

  // Extract clusters, remembering the semantic label each one was observed with.
  std::unordered_map<FrameData::ObjectImageType, Pixels> clusters;
  std::unordered_map<FrameData::ObjectImageType, std::unordered_map<int, int>> cluster_labels;
  for (int u = 0; u < cluster_image.cols; u++) {
    for (int v = 0; v < cluster_image.rows; v++) {
      const auto& id = cluster_image.at<FrameData::InstanceImageType>(v, u);
      if (id == 0) {
        continue;
      }
      // Instance masks may cover pixels for which the RGB-D sensor has no
      // geometric measurement. Never forward those pixels: zero-depth points
      // map to the camera origin and otherwise inflate the object bounding box
      // and its terminal private TSDF reconstruction.
      if (!isValidObjectMeasurementPixel(data.input, u, v, config.max_range)) {
        continue;
      }
      if (have_instances && !data.input.label_image.empty()) {
        ++cluster_labels[id][data.input.label_image.at<InputData::LabelType>(v, u)];
      }

      // Filter background based on given prompt
      if (background_) {
        const auto feature = data.input.label_features.find(id);
        if (feature == data.input.label_features.end()) {
          continue;
        }
        auto score = background_->getBestScore(*metric_, feature->second);
        if (score.score > config.max_background_score) {
          continue;
        }
      }

      data.object_image.at<FrameData::ObjectImageType>(v, u) = id;
      clusters[id].emplace_back(u, v);
    }
  }

  for (const auto& [id, pixels] : clusters) {
    const auto curr_num_pixels = static_cast<int>(pixels.size());
    if (curr_num_pixels < config.min_cluster_size ||
        (config.max_cluster_size > 0 && curr_num_pixels > config.max_cluster_size)) {
      continue;
    }

    MeasurementCluster cluster;
    cluster.pixels.insert(cluster.pixels.end(), pixels.begin(), pixels.end());
    cluster.id = id;

    if (filter_by_volume_) {
      const auto bbox = BoundingBox(utils::VertexMapAdaptor(cluster.pixels, data.input.vertex_map));
      const auto volume = bbox.volume();
      if (volume < config.min_object_volume ||
          (config.max_object_volume > 0.0 && volume > config.max_object_volume)) {
        continue;
      }
    }

    // Closed set version. With an instance channel the cluster id is the physical
    // instance, so the semantic class comes from the label image instead: take the
    // label the cluster's pixels carried most often.
    if (data.input.label_features.empty()) {
      int semantic_id = id;
      const auto labels = cluster_labels.find(id);
      if (labels != cluster_labels.end() && !labels->second.empty()) {
        semantic_id = std::max_element(labels->second.begin(),
                                       labels->second.end(),
                                       [](const auto& lhs, const auto& rhs) {
                                         return lhs.second < rhs.second;
                                       })
                          ->first;
      }
      cluster.semantics = SemanticClusterInfo(semantic_id);
    }
    // TODO(Yun) For now all semantic id is the same (so all label checks are invalid)
    const auto feature = data.input.label_features.find(id);
    if (feature != data.input.label_features.end()) {
      cluster.semantics = SemanticClusterInfo(id, feature->second);
    }

    data.semantic_clusters.emplace_back(std::move(cluster));
  }
}

}  // namespace khronos
