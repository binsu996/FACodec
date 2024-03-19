import facodec

# 获取音频的prosody编码、content编码以及speaker_embedding
codec=facodec.FACodec.from_pretrain().to("mps")
prosody_codes, content_codes, residual_codes, spk_embs=codec.get_codes_and_embedding_from_file("a.wav")
print(prosody_codes.shape)
print(content_codes.shape)
print(residual_codes.shape)
print(spk_embs.shape)

# 将编码转换为embedding
prosody_embedding=codec.prosody2emb(prosody_codes)
content_embedding=codec.prosody2emb(content_codes)

print(prosody_embedding.shape)
print(content_embedding.shape)

# 执行语音转换
vc=facodec.FACodecVC.from_pretrain().to("mps")
c_wav=vc("a.wav","b.wav")
print(c_wav.shape)
vc.save_audio(c_wav,"c.wav")