## FACodec的非官方封装

主要提供两个方便的接口来使用预训练模型
```python

# 提取codes和embedding
codec=facodec.FACodec.from_pretrain().to("mps")
codes=codec.get_codes_and_embedding_from_file("a.wav")
print(codes)

# 语音转换
vc=facodec.FACodecVC.from_pretrain().to("mps")
c_wav=vc("a.wav","b.wav")
print(c_wav)
vc.save_audio(c_wav,"c.wav")
```


###  安装
pip install git+https://github.com/binsu996/FACodec
