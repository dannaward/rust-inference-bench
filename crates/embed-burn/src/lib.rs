//! Burn implementation of [`InferenceEngine`].
//!
//! The model is generated at build time by burn-import from the all-MiniLM ONNX
//! export (see build.rs), then run on the ndarray (CPU) backend. NOT the
//! burn-candle backend — that would make the comparison Candle-vs-Candle.
//!
//! Pooling + normalization mirror the Candle engine exactly (attention-masked
//! mean pooling + L2 norm) so the two are directly comparable for the parity gate.

#![allow(clippy::all)]

mod minilm {
    include!(concat!(env!("OUT_DIR"), "/model/model.rs"));
}

use anyhow::{Context, Result};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Int, Tensor, TensorData};
use embed_core::{Embedding, InferenceEngine, EMBED_DIM};
use tokenizers::Tokenizer;

type B = NdArray<f32>;

const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";
const REVISION: &str = "main";

/// Burn-backed embedding engine (ndarray backend for the desktop CPU baseline).
pub struct BurnEngine {
    name: String,
    model: minilm::Model<B>,
    tokenizer: Tokenizer,
    device: NdArrayDevice,
}

impl BurnEngine {
    /// Load the burn-import-generated model (weights embedded at build time) and
    /// the tokenizer (fetched from the HF Hub, cached).
    pub fn load(_model_dir: &str) -> Result<Self> {
        use hf_hub::{api::sync::Api, Repo, RepoType};

        let device = NdArrayDevice::Cpu;
        let model = minilm::Model::<B>::default();

        let repo = Api::new()?.repo(Repo::with_revision(
            MODEL_ID.to_string(),
            RepoType::Model,
            REVISION.to_string(),
        ));
        let tokenizer_path = repo.get("tokenizer.json").context("fetch tokenizer.json")?;
        let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
        tokenizer
            .with_padding(Some(tokenizers::PaddingParams::default()))
            .with_truncation(Some(tokenizers::TruncationParams::default()))
            .map_err(anyhow::Error::msg)?;

        Ok(Self {
            name: "burn-ndarray".to_string(),
            model,
            tokenizer,
            device,
        })
    }

    fn forward(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(anyhow::Error::msg)?;

        let n = encodings.len();
        let seq = encodings[0].len();
        let mut ids = Vec::with_capacity(n * seq);
        let mut mask = Vec::with_capacity(n * seq);
        for enc in &encodings {
            ids.extend(enc.get_ids().iter().map(|&x| x as i64));
            mask.extend(enc.get_attention_mask().iter().map(|&x| x as i64));
        }

        let input_ids =
            Tensor::<B, 2, Int>::from_data(TensorData::new(ids, [n, seq]), &self.device);
        let attention =
            Tensor::<B, 2, Int>::from_data(TensorData::new(mask, [n, seq]), &self.device);
        let token_type = Tensor::<B, 2, Int>::zeros([n, seq], &self.device);

        // [n, seq, hidden]
        let hidden = self
            .model
            .forward(input_ids, attention.clone(), token_type);

        // Attention-masked mean pooling: sum(hidden * mask) / sum(mask).
        let mask_f = attention.float(); // [n, seq]
        let mask3 = mask_f.clone().unsqueeze_dim::<3>(2); // [n, seq, 1]
        let summed = hidden.mul(mask3).sum_dim(1); // [n, 1, hidden]
        let counts = mask_f.sum_dim(1).unsqueeze_dim::<3>(2); // [n, 1, 1]
        let mean = summed.div(counts); // [n, 1, hidden]

        // L2 normalize over the hidden dim.
        let norm = mean.clone().powf_scalar(2.0).sum_dim(2).sqrt(); // [n, 1, 1]
        let normalized = mean.div(norm).reshape([n, EMBED_DIM]); // [n, hidden]

        let flat: Vec<f32> = normalized
            .into_data()
            .to_vec()
            .map_err(|e| anyhow::anyhow!("tensor to_vec: {e:?}"))?;
        Ok(flat.chunks(EMBED_DIM).map(|c| c.to_vec()).collect())
    }
}

impl InferenceEngine for BurnEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn embed(&self, text: &str) -> Result<Embedding> {
        Ok(self.forward(&[text.to_string()])?.remove(0))
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Embedding>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        self.forward(texts)
    }
}
