from __future__ import annotations
from phase2_offline_edge_companion.hardware_sensors import (
    I2CBusAdapter,
    MAX30102Driver,
    MPU6050Driver,
    BMP280Driver,
    HardwareSensorBridge,
)


def test_i2c_adapter_emulation_graceful_fallback() -> None:
    bus = I2CBusAdapter(bus_number=99)
    assert bus.is_hardware_available is False
    assert bus.read_byte_data(0x57, 0x00) == 0
    assert bus.read_i2c_block_data(0x68, 0x00, 6) == [0] * 6
    bus.write_byte_data(0x76, 0x00, 0x55)
    bus.close()


def test_drivers_initialize_safely_when_offline() -> None:
    bus = I2CBusAdapter(bus_number=99)
    max_d = MAX30102Driver(bus)
    mpu_d = MPU6050Driver(bus)
    bmp_d = BMP280Driver(bus)

    assert max_d.is_connected is False
    assert max_d.read_fifo_samples(max_samples=10) == []

    assert mpu_d.is_connected is False
    assert mpu_d.read_accel_g() == (0.0, 0.0, 1.0)

    assert bmp_d.is_connected is False
    assert bmp_d.read_pressure_hpa() == 1013.25


def test_hardware_sensor_bridge_status_and_read_packet() -> None:
    bridge = HardwareSensorBridge(bus_number=99)
    st = bridge.get_hardware_status()
    assert "hardware_present" in st
    assert "max30102_ppg" in st["sensors"]
    assert "mpu6050_imu" in st["sensors"]
    assert "bmp280_baro" in st["sensors"]

    packet = bridge.read_packet(fallback_alt_m=3200.0, activity_mode="hiking")
    assert "max30102" in packet
    assert "mpu6050" in packet
    assert "bmp280" in packet
    assert packet["bmp280"]["altitude_m"] > 2000.0
    bridge.close()
