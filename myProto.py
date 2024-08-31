'''
proto with label, core, radio, expected output

理论支撑：
通过高斯噪声产生的proto-output 与 原始数据产生的data-output对 是否具有一致性
如果证明二者一致，可以通过保存proto-output而不是模型节省存储空间和推理时间
即使不一致，也可以直接保存data-output，而不是模型
'''
class myProto:
    def __init__(self,label,output,core=None,radio=None,data=None):
        self.label=label
        self.core=core
        self.radio=radio
        self.output=output
        self.data=data
    
    def get_data(self):
        pass
