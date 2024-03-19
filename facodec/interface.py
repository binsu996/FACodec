from .facodec import FACodecDecoder, FACodecDecoderV2, FACodecEncoder, FACodecEncoderV2, FactorizedVectorQuantize, FACodecRedecoder
from huggingface_hub import hf_hub_download
import torch
from torch import nn
import librosa
import soundfile as sf


class FACodec(nn.Module):
    def __init__(self, fa_encoder: FACodecEncoder, fa_decoder: FACodecDecoder) -> None:
        super().__init__()
        self.fa_encoder = fa_encoder
        self.fa_decoder = fa_decoder

    @classmethod
    def from_pretrain(cls):
        fa_encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )

        fa_decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )

        encoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
        decoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

        fa_encoder.load_state_dict(torch.load(encoder_ckpt))
        fa_decoder.load_state_dict(torch.load(decoder_ckpt))

        fa_encoder.eval()
        fa_decoder.eval()

        return FACodec(fa_encoder, fa_decoder)
    
    def to(self,device):
        super().to(device)
        self.device=device
        return self

    def load_audio(self, audio_path):
        wav = librosa.load(audio_path, sr=16000, mono=True)[0]
        wav = torch.from_numpy(wav).float().to(self.device)
        return wav

    @torch.no_grad()
    def get_codes_and_embedding(self, wav):
        wav = wav.unsqueeze(0).unsqueeze(0)
        enc_out = self.fa_encoder(wav)
        vq_post_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(
            enc_out, eval_vq=False, vq=True)
        prosody_code = vq_id[:1]
        content_code = vq_id[1:3]
        residual_code = vq_id[3:]
        return prosody_code, content_code, residual_code, spk_embs

    def get_codes_and_embedding_from_file(self, audio_path):
        wav = self.load_audio(audio_path)
        return self.get_codes_and_embedding(wav)
    
    def prosody2emb(self,prosody_codes):
        return self.fa_decoder.prosody2emb(prosody_codes)

    def content2emb(self,content_codes):
        return self.fa_decoder.content2emb(content_codes)


class FACodecVC(nn.Module):
    def __init__(self, fa_codec: FACodec, fa_redecoder: FACodecRedecoder) -> None:
        super().__init__()
        self.fa_codec = fa_codec
        self.fa_redecoder = fa_redecoder

    @classmethod
    def from_pretrain(cls):
        fa_redecoder = FACodecRedecoder()
        redecoder_ckpt = hf_hub_download(
            repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_redecoder.bin")
        fa_redecoder.load_state_dict(torch.load(redecoder_ckpt))
        fa_redecoder = fa_redecoder

        fa_codec = FACodec.from_pretrain()
        return FACodecVC(fa_codec, fa_redecoder)

    @torch.no_grad()
    def convert(self, speaker_from, content_to):
        wav_a = self.fa_codec.load_audio(speaker_from)
        wav_b = self.fa_codec.load_audio(content_to)
        wav_a = wav_a.unsqueeze(0).unsqueeze(0)
        wav_b = wav_b.unsqueeze(0).unsqueeze(0)
        enc_out_a = self.fa_codec.fa_encoder(wav_a)
        enc_out_b = self.fa_codec.fa_encoder(wav_b)

        vq_post_emb_a, vq_id_a, _, quantized_a, spk_embs_a = self.fa_codec.fa_decoder(
            enc_out_a, eval_vq=False, vq=True)
        vq_post_emb_b, vq_id_b, _, quantized_b, spk_embs_b = self.fa_codec.fa_decoder(
            enc_out_b, eval_vq=False, vq=True)

        vq_post_emb_a_to_b = self.fa_redecoder.vq2emb(
            vq_id_b, spk_embs_a, use_residual=False)
        recon_wav_a_to_b = self.fa_redecoder.inference(
            vq_post_emb_a_to_b, spk_embs_a)
        return recon_wav_a_to_b

    def save_audio(self, wav, filename):
        sf.write(filename, wav.cpu().numpy().squeeze(), 16000, 'PCM_16')

    def to(self,device):
        super().to(device)
        self.fa_codec.to(device)
        self.device=device
        return self

    def forward(self, speaker_from, content_to):
        return self.convert(speaker_from, content_to)

