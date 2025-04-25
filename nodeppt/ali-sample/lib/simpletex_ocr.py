# SrjmD4kjauTgUS9pWrObt93eSVPOv4dqZF8uhF5Ma8Si9748wjPd9x4MWlo6KbXs
# appid 3Uf1eLy4XuXzdjLKXrruIjPc
# appsecret ymlDrnMeZs137yMXFKz8Ht1eYCL2982e




import datetime
import json
import requests
from random import Random
import hashlib


class SimpleTexOcr:
    SIMPLETEX_APP_ID = "3Uf1eLy4XuXzdjLKXrruIjPc"
    SIMPLETEX_APP_SECRET = "ymlDrnMeZs137yMXFKz8Ht1eYCL2982e"

    def __init__(self):
        
        pass

    @staticmethod
    def random_str(randomlength=16):
        str = ''
        chars = 'AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789'
        length = len(chars) - 1
        random = Random()
        for i in range(randomlength):
            str += chars[random.randint(0, length)]
        return str


    def get_req_data(self, req_data, appid, secret):
        header = {}
        header["timestamp"] = str(int(datetime.datetime.now().timestamp()))
        header["random-str"] = self.random_str(16)
        header["app-id"] = appid
        pre_sign_string = ""
        sorted_keys = list(req_data.keys()) + list(header)
        sorted_keys.sort()
        for key in sorted_keys:
            if pre_sign_string:
                pre_sign_string += "&"
            if key in header:
                pre_sign_string += key + "=" + str(header[key])
            else:
                pre_sign_string += key + "=" + str(req_data[key])

        pre_sign_string += "&secret=" + secret
        header["sign"] = hashlib.md5(pre_sign_string.encode()).hexdigest()
        return header, req_data

    def query(self, img_path):
        img_file = {"file": open(img_path, 'rb')}
        data = {}
        # 请求参数数据（非文件型参数），视情况填入，可以参考各个接口的参数说明
        header, data = self.get_req_data(data, self.SIMPLETEX_APP_ID, self.SIMPLETEX_APP_SECRET)
        res = requests.post("https://server.simpletex.cn/api/latex_ocr", files=img_file, data=data, headers=header)
        return json.loads(res.text)

pass