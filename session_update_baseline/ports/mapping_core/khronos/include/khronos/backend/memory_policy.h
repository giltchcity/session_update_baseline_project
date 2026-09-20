#pragma once

namespace khronos {

/**
 * @brief Ablation switches for the cross-session memory policy. Set once from
 * Backend::Config before the first reconciliation and read-only afterwards, so
 * every run records in config.txt which policy produced its map.
 *
 * The default is the method as published: inherited surface that this session
 * re-observed and disagrees with beyond the reconstruction resolution is
 * superseded by the present measurement.
 */
struct MemoryPolicy {
  // Agreement scale for one physical state's inherited and session surface, in
  // units of the object surface resolution.
  float object_agreement_voxels = 0.5f;

  // Agreement scale for inherited background surface, in units of the map voxel.
  float background_agreement_voxels = 0.5f;

  // If false, geometric disagreement never retires inherited surface: memory is
  // removed only where this session's rays measured it absent. This is the
  // parameter-free policy.
  bool retire_disagreeing_memory = true;

  // If false, inherited memory is not rigidly registered into this session's
  // frame before it is compared.
  bool register_inherited_memory = true;
};

MemoryPolicy& mutableMemoryPolicy();
const MemoryPolicy& memoryPolicy();

void declare_config(MemoryPolicy& config);

}  // namespace khronos
