"""
DT-2000 设备数据采集demo代码
适配硬件手册V1.2
功能：串口连接设备、数据读取、异常重连
"""
import serial
import time


class DeviceCollector:
    def __init__(self):
        # 根据手册默认波特率 115200
        self.baud_rate = 115200
        self.ser = None

    def connect(self, port="COM3"):
        """连接DT2000采集设备"""
        try:
            self.ser = serial.Serial(port, self.baud_rate, timeout=1)
            print("设备连接成功")
            return True
        except Exception as e:
            print("设备连接失败：", e)
            return False

    def read_data(self):
        """读取设备8通道采集数据"""
        if self.ser is None or not self.ser.is_open:
            return "设备未连接"

        self.ser.write(b"READ\r\n")
        data = self.ser.readline()
        return data.decode("utf-8")

    def close(self):
        """关闭设备连接"""
        if self.ser:
            self.ser.close()


# 主运行函数
if __name__ == "__main__":
    collect = DeviceCollector()
    if collect.connect():
        for i in range(5):
            res = collect.read_data()
            print("采集数据：", res)
            time.sleep(0.5)
        collect.close()
