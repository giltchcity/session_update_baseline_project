#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "fastdds_session_update_ack.xml"
RUNNER = ROOT / "scripts" / "run_khronos_session_strict.sh"
NS = {"f": "http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles"}
TRANSACTION_WRITER_TOPICS = [
    "/session_update/frame_processed",
    "/nss/rgb/image_raw",
    "/nss/depth/image_raw",
    "/nss/semantic/image_raw",
]


def duration_ns(parent: ET.Element, name: str) -> int:
    value = parent.find(f"f:{name}", NS)
    if value is None:
        raise AssertionError(f"missing {name}")
    seconds = int(value.findtext("f:sec", "0", NS))
    nanoseconds = int(value.findtext("f:nanosec", "0", NS))
    return seconds * 1_000_000_000 + nanoseconds


def main() -> None:
    root = ET.parse(PROFILE).getroot()
    publishers = root.findall("f:profiles/f:publisher", NS)
    topic_profiles = [
        item.attrib.get("profile_name", "")
        for item in publishers
        if not item.attrib.get("is_default_profile") == "true"
    ]
    assert topic_profiles == TRANSACTION_WRITER_TOPICS, topic_profiles
    tuned = {
        item.attrib["profile_name"]: item
        for item in publishers
        if item.attrib.get("profile_name") in TRANSACTION_WRITER_TOPICS
    }
    assert list(tuned) == TRANSACTION_WRITER_TOPICS
    for topic in TRANSACTION_WRITER_TOPICS:
        times = tuned[topic].find("f:times", NS)
        assert times is not None, f"missing writer times: {topic}"
        assert duration_ns(times, "initialHeartbeatDelay") == 1_000_000, topic
        assert duration_ns(times, "heartbeatPeriod") == 10_000_000, topic
        assert duration_ns(times, "nackResponseDelay") == 1_000_000, topic

    # Turning XML QoS on must not silently change the large image endpoints.
    defaults = [
        item
        for item in publishers
        if item.attrib.get("is_default_profile") == "true"
    ]
    subscribers = root.findall("f:profiles/f:subscriber", NS)
    assert len(defaults) == 1 and len(subscribers) == 1
    assert subscribers[0].attrib.get("is_default_profile") == "true"
    for item in (defaults[0], *tuned.values(), subscribers[0]):
        assert item.findtext("f:historyMemoryPolicy", "", NS) == (
            "PREALLOCATED_WITH_REALLOC"
        )
        assert item.findtext("f:qos/f:data_sharing/f:kind", "", NS) == "OFF"
    for item in (defaults[0], *tuned.values()):
        assert item.findtext("f:qos/f:publishMode/f:kind", "", NS) == (
            "SYNCHRONOUS"
        )

    runner = RUNNER.read_text(encoding="utf-8")
    assert "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" in runner
    assert "export FASTRTPS_DEFAULT_PROFILES_FILE=" in runner
    assert "export RMW_FASTRTPS_USE_QOS_FROM_XML=1" in runner
    assert "unset FASTDDS_DEFAULT_PROFILES_FILE" in runner
    assert "export FASTDDS_DEFAULT_PROFILES_FILE=" not in runner
    assert "fastdds_session_update_ack.xml" in runner
    assert '"${BASE1_BUILD_DIR}/test_fastdds_ack_profile"' in runner
    print(
        "FAST_DDS_TRANSACTION_XML_CONTRACT_OK "
        f"profile={PROFILE.resolve()} topics={','.join(TRANSACTION_WRITER_TOPICS)}"
    )


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ET.ParseError) as error:
        print(f"FAST_DDS_TRANSACTION_XML_CONTRACT_ERROR {error}", file=sys.stderr)
        raise
