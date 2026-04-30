# Audio & Speech Models

*Depth files for speech encoders, self-supervised speech pretraining, and the integration patterns that bolt audio onto LLMs.*

---

## Reading order

1. **[mel-spectrogram](mel-spectrogram.md)** — the canonical input representation for speech: STFT → mel filterbank → log. 80 channels, 10 ms stride.
2. **[conformer](conformer.md)** — the convolution-augmented Transformer that is the default speech-encoder architecture since 2020.
3. **[whisper](whisper.md)** — OpenAI's 680K-hour-trained encoder-decoder ASR / translation / language-ID model. The most-used open speech system.
4. **[wav2vec2](wav2vec2.md)** — the self-supervised contrastive pretraining that started the "speech BERT" era.
5. **[hubert](hubert.md)** — the masked-prediction alternative to wav2vec 2.0; k-means-based pseudo-labels.
6. **[best-rq](best-rq.md)** — the frozen-random-projection quantizer variant; simpler than wav2vec2/HuBERT, used by Llama 3 and Kimi k1.5.

## Related

- [vision/](../vision/) — analogous ladder on the image side.
- [_multimodal-fusion](../_multimodal-fusion.md) — taxonomy of integration patterns.
