/**
 * Python access to a saved session state (.4dmap).
 *
 * The RGB viewer is Python and the map is a C++ binary, which until now meant
 * exporting every time step to PLY before anything could be drawn. That detour
 * is what these bindings remove: the map is opened once and queried at any
 * stamp, the same way the C++ spatio-temporal visualizer does it.
 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <spark_dsg/dynamic_scene_graph.h>

#include "khronos/spatio_temporal_map/spatio_temporal_map.h"

namespace py = pybind11;

namespace {

/// One session state on disk, queryable at any of its stored stamps.
class SessionMap {
 public:
  explicit SessionMap(const std::string& path) {
    map_ = khronos::SpatioTemporalMap::load(path);
    if (!map_) {
      throw std::runtime_error("failed to load session map: " + path);
    }
  }

  size_t numTimeSteps() const { return map_->numTimeSteps(); }

  std::vector<uint64_t> stamps() const {
    std::vector<uint64_t> out;
    out.reserve(map_->numTimeSteps());
    for (size_t i = 0; i < map_->numTimeSteps(); ++i) {
      out.push_back(map_->getTimeStamp(i));
    }
    return out;
  }

  uint64_t earliest() const { return map_->earliest(); }
  uint64_t latest() const { return map_->latest(); }

  /// Background mesh at `stamp`, as numpy arrays.
  py::dict meshAt(uint64_t stamp) const {
    const auto dsg = map_->getDsgPtr(stamp);
    if (!dsg || !dsg->hasMesh()) {
      throw std::runtime_error("no mesh at requested stamp");
    }
    const auto& mesh = *dsg->mesh();
    const size_t nv = mesh.numVertices();
    const size_t nf = mesh.numFaces();

    py::array_t<float> positions({nv, size_t(3)});
    auto pos = positions.mutable_unchecked<2>();
    py::array_t<uint8_t> colors({nv, size_t(3)});
    auto col = colors.mutable_unchecked<2>();
    py::array_t<int32_t> labels(nv);
    auto lab = labels.mutable_unchecked<1>();
    const bool has_colors = mesh.has_colors;
    const bool has_labels = mesh.has_labels;

    for (size_t i = 0; i < nv; ++i) {
      const auto& p = mesh.pos(i);
      pos(i, 0) = p.x();
      pos(i, 1) = p.y();
      pos(i, 2) = p.z();
      if (has_colors) {
        const auto& c = mesh.color(i);
        col(i, 0) = c.r;
        col(i, 1) = c.g;
        col(i, 2) = c.b;
      } else {
        col(i, 0) = col(i, 1) = col(i, 2) = 200;
      }
      lab(i) = has_labels ? static_cast<int32_t>(mesh.label(i)) : -1;
    }

    py::array_t<uint32_t> faces({nf, size_t(3)});
    auto fac = faces.mutable_unchecked<2>();
    for (size_t i = 0; i < nf; ++i) {
      const auto& f = mesh.face(i);
      fac(i, 0) = static_cast<uint32_t>(f[0]);
      fac(i, 1) = static_cast<uint32_t>(f[1]);
      fac(i, 2) = static_cast<uint32_t>(f[2]);
    }

    py::dict out;
    out["positions"] = positions;
    out["colors"] = colors;
    out["labels"] = labels;
    out["faces"] = faces;
    return out;
  }

  /// Objects present at `stamp`: physical id, semantic label, bbox, mesh size.
  py::list objectsAt(uint64_t stamp) const {
    const auto dsg = map_->getDsgPtr(stamp);
    py::list out;
    if (!dsg || !dsg->hasLayer(spark_dsg::DsgLayers::OBJECTS)) {
      return out;
    }
    for (const auto& [node_id, node] : dsg->getLayer(spark_dsg::DsgLayers::OBJECTS).nodes()) {
      const auto* attrs = dynamic_cast<const spark_dsg::KhronosObjectAttributes*>(
          &node->attributes());
      if (!attrs) {
        continue;
      }
      py::dict entry;
      entry["node_id"] = static_cast<uint64_t>(node_id);
      entry["semantic_label"] = static_cast<int>(attrs->semantic_label);
      const auto found = attrs->details.find("instance_id");
      entry["physical_id"] =
          (found != attrs->details.end() && !found->second.empty())
              ? static_cast<int>(found->second.front())
              : 0;
      const auto& c = attrs->bounding_box.world_P_center;
      entry["center"] = std::vector<float>{c.x(), c.y(), c.z()};
      const auto d = attrs->bounding_box.dimensions;
      entry["dimensions"] = std::vector<float>{d.x(), d.y(), d.z()};
      entry["mesh_vertices"] = static_cast<int>(attrs->mesh.numVertices());
      out.append(entry);
    }
    return out;
  }

 private:
  std::shared_ptr<khronos::SpatioTemporalMap> map_;
};

}  // namespace

PYBIND11_MODULE(session_map, m) {
  m.doc() = "Read a saved session state (.4dmap) directly from Python.";
  py::class_<SessionMap>(m, "SessionMap")
      .def(py::init<const std::string&>(), py::arg("path"))
      .def_property_readonly("num_time_steps", &SessionMap::numTimeSteps)
      .def_property_readonly("stamps", &SessionMap::stamps)
      .def_property_readonly("earliest", &SessionMap::earliest)
      .def_property_readonly("latest", &SessionMap::latest)
      .def("mesh_at", &SessionMap::meshAt, py::arg("stamp"))
      .def("objects_at", &SessionMap::objectsAt, py::arg("stamp"));
}
