import facodec

codec=facodec.FACodec.from_pretrain().to("mps")
codes=codec.get_codes_and_embedding_from_file("a.wav")
print(codes)

vc=facodec.FACodecVC.from_pretrain().to("mps")
c_wav=vc("a.wav","b.wav")
print(c_wav)
vc.save_audio(c_wav,"c.wav")