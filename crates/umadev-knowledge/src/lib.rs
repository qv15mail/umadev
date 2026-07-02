//! `umadev-knowledge` — the structured knowledge base that replaces the
//! old "one folder + keyword sort + dump into prompt" approach.
//!
//! UmaDev ships a curated `knowledge/` directory (domain standards,
//! playbooks, expert methodology, design systems). Until 4.5 this corpus
//! was consumed by a naive keyword-scoring path matcher
//! (`score_path` in `umadev-agent::phases`) that:
//! - only tokenised ASCII words ≥ 3 chars,
//! - matched against file paths (×2) + first 500 chars of content,
//! - fell back to dictionary order when no keyword overlapped
//!   (the common case for CJK requirements against English filenames).
//!
//! That produced irrelevant retrievals and, for Chinese requirements,
//! effectively a random sample. This crate does retrieval properly:
//!
//! - **`chunker`** — markdown-aware segmentation that strips front-matter
//!   and splits on `## H2` sections, preserving per-chunk metadata
//!   (path, title, tags, domain). One knowledge file becomes N chunks.
//! - **`index`** — a pure-Rust BM25 inverted index over the chunk corpus,
//!   with mixed ASCII + CJK-bigram tokenisation. Serialised to
//!   `.umadev/kb-index/bm25.bin` and rebuilt only when source files
//!   change (mtime-checked).
//! - **`vector`** — an OPTIONAL semantic layer. When `OPENAI_EMBED_KEY` is
//!   present it embeds each chunk via the `/v1/embeddings` endpoint the
//!   user's existing subscription already covers (no new dependency) and
//!   stores the vectors. No key → falls back to BM25 silently.
//! - **`retrieve`** — one entry point that picks the configured engine and
//!   returns ranked [`Chunk`] hits with scores, ready for the agent to
//!   format into a prompt or a TUI panel.
//!
//! ## Safety contract
//! - Pure functions over on-disk data. Retrieval NEVER blocks the
//!   pipeline: a corrupt/missing index returns an empty result, not an
//!   error (fail-open, same as the governance kernel).
//! - The optional vector layer touches the network only when explicitly
//!   enabled by an env var; the default engine is fully offline.

#![forbid(unsafe_code)]
#![warn(missing_docs, clippy::all, clippy::pedantic)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::doc_markdown,
    clippy::must_use_candidate,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::format_push_string,
    clippy::case_sensitive_file_extension_comparisons,
    clippy::map_unwrap_or,
    clippy::match_same_arms,
    clippy::unnecessary_map_or,
    clippy::unused_async,
    clippy::needless_continue,
    clippy::manual_let_else,
    clippy::nonminimal_bool
)]

pub mod chunker;
pub mod index;
/// Bundled local embedding backend (candle, pure Rust). Only compiled with the
/// `vector-local` feature; the npm package ships the model so it works offline.
#[cfg(feature = "vector-local")]
pub mod local_embed;
pub mod repomap;
pub mod retrieve;
pub mod tokenizer;
/// Per-chunk usefulness prior — retrieval-quality feedback that self-tunes the
/// curated-knowledge ranking from build outcomes (fail-open, bounded).
pub mod usefulness;
pub mod vector;

pub use chunker::{chunk_file, chunk_text, Chunk, ChunkMeta};
pub use index::{
    bm25_search, build_index, build_vector_store_if_enabled, invalidate_cache, load_or_build_index,
    load_or_build_index_multi, Bm25Index, Posting,
};
pub use repomap::{
    invalidate_cache as invalidate_repomap_cache, repo_map, symbol_index, FileSymbols, Symbol,
    SymbolIndex, SymbolKind, REPOMAP_CACHE_DIR,
};
pub use retrieve::{
    corpus_dirs, retrieve, retrieve_for_phase, retrieve_for_phase_with_expansion,
    retrieve_for_phase_with_vector, retrieve_with_vector, retrieve_with_vector_and_expansion,
    RetrievalConfig, RetrievalEngine, ScoredChunk,
};
pub use tokenizer::{cjk_trigrams_only, tokenize, tokenize_trigram};
pub use usefulness::{record_chunk_outcomes, record_chunk_outcomes_in, UsefulnessStore};
pub use vector::VectorStore;

/// Knowledge base index storage location, relative to the project root.
/// The BM25 index (`bm25.bin`) and optional vector store (`vectors.bin`)
/// live here; both are created on demand by their writers.
pub const KB_INDEX_DIR: &str = ".umadev/kb-index";

/// Test-only support: serialise + isolate tests that mutate the process-global
/// embedding env vars (`UMADEV_EMBED_DIM` / `UMADEV_EMBED_MODEL` / the
/// `OPENAI_*` keys / `UMADEV_EMBED_MODEL_DIR`). Rust runs a crate's tests in
/// parallel threads sharing one process, so without serialisation these tests
/// race on shared state — and, under the `vector-local` feature, a model
/// installed at `~/.umadev/embed-model` would otherwise make every "no backend"
/// test see a live local embedder.
#[cfg(test)]
pub(crate) mod testsupport {
    use std::sync::{Mutex, MutexGuard, OnceLock};

    fn lock() -> &'static Mutex<()> {
        static L: OnceLock<Mutex<()>> = OnceLock::new();
        L.get_or_init(|| Mutex::new(()))
    }

    /// Acquire the process-wide env lock so env-mutating tests don't race.
    /// Poison-tolerant: a panicking holder doesn't cascade into the next test.
    pub fn env_guard() -> MutexGuard<'static, ()> {
        lock()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// RAII guard that holds the env lock AND points the bundled local-embed
    /// backend at an empty directory, so `local_embed::is_available()` is false
    /// regardless of any real model installed at `~/.umadev/embed-model`. The
    /// previous `UMADEV_EMBED_MODEL_DIR` value is restored on drop.
    pub struct NoLocalModel {
        _guard: MutexGuard<'static, ()>,
        _dir: tempfile::TempDir,
        prev: Option<String>,
    }

    impl Drop for NoLocalModel {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("UMADEV_EMBED_MODEL_DIR", v),
                None => std::env::remove_var("UMADEV_EMBED_MODEL_DIR"),
            }
        }
    }

    /// Neutralise the local backend for the duration of a test (see
    /// [`NoLocalModel`]). Returns a guard that must be kept alive.
    #[must_use]
    pub fn without_local_model() -> NoLocalModel {
        let guard = env_guard();
        let prev = std::env::var("UMADEV_EMBED_MODEL_DIR").ok();
        let dir = tempfile::TempDir::new().expect("tempdir");
        std::env::set_var("UMADEV_EMBED_MODEL_DIR", dir.path());
        NoLocalModel {
            _guard: guard,
            _dir: dir,
            prev,
        }
    }
}
