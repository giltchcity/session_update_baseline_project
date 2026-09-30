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

#include "khronos/active_window/object_extraction/object_worker_pool.h"

#include <limits>
#include <stdexcept>

#include <config_utilities/config.h>

namespace khronos {

using namespace std::chrono_literals;
using hydra::timing::ElapsedTimeRecorder;
using spark_dsg::NodeAttributes;

void declare_config(ObjectWorkerPool::Config& config) {
  using namespace config;
  name("ObjectWorkerPool::Config");
  field(config.num_workers, "num_workers");
  field(config.poll_time_us, "poll_time_us");
  field(config.verbosity, "verbosity");
}

ObjectWorkerPool::Request::Request(uint64_t generation,
                                   TimeStamp stamp,
                                   const Track& track,
                                   const FrameDataBuffer& frame_data)
    : generation(generation), stamp(stamp), track(track), frame_data(frame_data) {}

ObjectWorkerPool::ObjectWorkerPool(const Config& config,
                                   std::unique_ptr<ObjectExtractor>&& extractor)
    : config(config),
      extractor_(std::move(extractor)),
      source_scope_(session_detail::makeExtractionScope()) {
  // ElapsedTimeRecorder owns a lazily-created process singleton whose instance()
  // accessor is not itself synchronized. Initialize it on the constructing
  // thread before two extraction workers can reach record() concurrently.
  (void)ElapsedTimeRecorder::instance();
  if (extractor_) {
    spin_thread_ = std::make_unique<std::thread>(&ObjectWorkerPool::spin, this);
  }
}

ObjectWorkerPool::~ObjectWorkerPool() { stop(); }

void ObjectWorkerPool::stop() {
  std::lock_guard<std::mutex> shutdown_lock(stop_mutex_);
  uint64_t target_generation = 0;
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    accepting_ = false;
    target_generation = accepted_generation_;
  }

  // Every request must finish its last access to the pool before its state is
  // destroyed. The generation frontier also includes requests that the
  // dispatcher has not taken from work_queue_ yet.
  {
    std::unique_lock<std::mutex> lock(state_mutex_);
    state_cv_.wait(lock, [this, target_generation] {
      return completions_.completedThrough() >= target_generation;
    });
    stopping_ = true;
  }
  state_cv_.notify_all();

  if (spin_thread_) {
    spin_thread_->join();
  }

  spin_thread_.reset();
}

void ObjectWorkerPool::join() {
  uint64_t target_generation = 0;
  std::exception_ptr failure;
  {
    std::unique_lock<std::mutex> lock(state_mutex_);
    target_generation = accepted_generation_;
    CLOG(5) << "Waiting for object extraction generation " << target_generation
            << " (completed through " << completions_.completedThrough()
            << ", workers " << curr_workers_ << ").";
    state_cv_.wait(lock, [this, target_generation] {
      return completions_.completedThrough() >= target_generation;
    });
    failure = completions_.failure();
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
}

size_t ObjectWorkerPool::numRunning() const {
  std::lock_guard<std::mutex> lock(state_mutex_);
  return curr_workers_;
}

void ObjectWorkerPool::submit(TimeStamp stamp,
                              const Track& track,
                              const FrameDataBuffer& frame_data) {
  if (!extractor_) {
    return;
  }

  // Copy the immutable input before accepting a generation. Allocation or
  // enqueue failure must not leave an accepted slot that can never complete.
  auto request = std::make_shared<Request>(0, stamp, track, frame_data);
  std::lock_guard<std::mutex> lock(state_mutex_);
  if (!accepting_) {
    throw std::logic_error("cannot submit object extraction after worker pool shutdown");
  }
  completions_.rethrowFailure();
  if (accepted_generation_ == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("Exhausted object extraction generations");
  }
  request->generation = accepted_generation_ + 1;
  completions_.reserve(request->generation);
  try {
    if (!work_queue_.push(request)) {
      throw std::runtime_error("Object extraction request was not queued");
    }
  } catch (...) {
    completions_.cancelReservation(request->generation);
    throw;
  }
  accepted_generation_ = request->generation;
}

KhronosObjectAttributes::Ptr ObjectWorkerPool::runBlocking(const Track& track,
                                                           const FrameDataBuffer& data) const {
  if (!extractor_) {
    return nullptr;
  }

  return extractor_->extractObject(track, data);
}

void ObjectWorkerPool::fill(hydra::LayerUpdate& update) {
  std::lock_guard<std::mutex> lock(state_mutex_);
  auto ready = completions_.takeReady();
  update.attributes.splice(update.attributes.end(), ready);
}

void ObjectWorkerPool::spin() {
  if (!extractor_) {
    return;
  }

  while (true) {
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (stopping_) {
        return;
      }
    }
    if (!work_queue_.poll(config.poll_time_us)) {
      continue;
    }

    {
      std::unique_lock<std::mutex> lock(state_mutex_);
      state_cv_.wait(lock, [this] {
        return stopping_ || !config.num_workers || curr_workers_ < config.num_workers;
      });
      if (stopping_) {
        return;
      }
      ++curr_workers_;
    }

    auto request = work_queue_.pop();
    const auto generation = request->generation;
    std::thread worker;
    try {
      worker = std::thread(&ObjectWorkerPool::runOnce, this, std::move(request));
    } catch (...) {
      markCompleted(generation, nullptr, std::current_exception());
      continue;
    }
    try {
      worker.detach();
    } catch (...) {
      // A constructed worker completes its own slot. Retain and join it if
      // detaching fails, rather than destroying a joinable temporary thread.
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        completions_.recordFailure(std::current_exception());
      }
      worker.join();
    }
  }
}

void ObjectWorkerPool::runOnce(Request::Ptr req) {
  const auto start = std::chrono::high_resolution_clock::now();
  spark_dsg::NodeAttributes::Ptr attrs;
  std::exception_ptr failure;
  try {
    auto extracted = extractor_->extractObject(req->track, req->frame_data);
    if (extracted) {
      session_detail::setExtractionSource(
          *extracted, {source_scope_, req->generation, req->stamp});
    }
    attrs = std::move(extracted);
  } catch (...) {
    failure = std::current_exception();
  }
  const auto stop = std::chrono::high_resolution_clock::now();

  try {
    ElapsedTimeRecorder::instance().record("active_window/extract_object", req->stamp, stop - start);
  } catch (...) {
    if (!failure) failure = std::current_exception();
  }
  markCompleted(req->generation, std::move(attrs), failure);
}

}  // namespace khronos
