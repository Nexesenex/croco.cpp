#include "arg.h"
#include "ggml.h"
#include "common.h"
#include "ngram-cache.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <vector>

int main(int argc, char ** argv){
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_LOOKUP)) {
        return 1;
    }

    common_init();

    // max. number of additional tokens to draft if match is found
    const int n_draft = params.n_draft;
    const int n_seq   = std::max(n_draft, 1);

    const bool dump_kv_cache = params.dump_kv_cache;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    // load the model
    common_init_result llama_init = common_init_from_params(params);

    llama_model * model = llama_init.model;
    llama_context * ctx = llama_init.context;

    // tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx, params.prompt, true, true);

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;
    int64_t t_draft_flat_us = 0;
    int64_t t_draft_us = 0;

    {
        // Fill up context ngram cache with tokens from user input:
        const int64_t t_start_draft_us = ggml_time_us();
        common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, inp.size(), false);

        if (!params.lookup_cache_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(params.lookup_cache_static);
            } catch (std::ifstream::failure const &) {
                LOG_ERR("failed to open static lookup cache: %s", params.lookup_cache_static.c_str());
                exit(1);
            }
        }

        if (!params.lookup_cache_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(params.lookup_cache_dynamic);
            } catch (std::ifstream::failure const &) {} // if the file does not exist it will simply be created at the end of the program
        }

        t_draft_flat_us += ggml_time_us() - t_start_draft_us;
    }

    const int max_context_size     = llama_n_ctx(ctx);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx, id).c_str());
    }

    fflush(stderr);

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    llama_decode(ctx, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx, llama_batch_get_one(&inp.back(),           1));

    const auto t_enc_end = ggml_time_us();

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past = inp.size();

    bool has_eos = false;

    struct common_sampler * smpl = common_sampler_init(model, params.sparams);

    std::vector<llama_draft_t> drafts;

    common_batch batch_tgt = llama_batch_init(max_context_size, 0, n_seq);
    std::vector<std::vector<int>> sampling_idx_store;
    sampling_idx_store.resize(n_seq);
    sampling_idx_store[0].push_back(0);

    // debug
    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, 1);

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // debug
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            common_kv_cache_dump_view_seqs(kvc_view, 40);
        }

        int i_dft = 0;
        int seq_best = 0;
        while (true) {
            // sample from the target model
            llama_token id = common_sampler_sample(ctx_sampling, ctx, NULL, sampling_idx_store[seq_best][i_dft]);

            common_sampler_accept(smpl, id, true);

            const std::string token_str = common_token_to_piece(ctx, id);

            if (!params.use_color) {
                LOG("%s", token_str.c_str());
            }

            if (llama_token_is_eog(model, id)) {
                has_eos = true;
            }

            ++n_predict;

            // check if the target token matches the draft
            bool accepted = false;
            for (int j = 0; j < (int) drafts.size() && !has_eos && !drafts.empty(); ++j) {
                if (i_dft + 1 < (int) drafts[j].size() && id == drafts[j][i_dft + 1]) {
                    LOG("draft success: (%d, '%s'), seq_id=%d\n", id, token_str.c_str(), j);
                    ++n_accept;
                    ++n_past;
                    ++i_dft;
                    inp.push_back(id);
                    seq_best = j;
                    {
                        // Update context ngram cache with the newly accepted token:
                        const int64_t t_start_draft_us = ggml_time_us();
                        common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
                        t_draft_us += ggml_time_us() - t_start_draft_us;
                    }

                    if (params.use_color) {
                        // color accepted draft token
                        printf("\033[34m%s\033[0m", token_str.c_str());
                        fflush(stdout);
                    }
                    accepted = true;
                    break;
                }
            }
            if (accepted) {
                continue;
            }

            if (params.use_color) {
                LOG("%s", token_str.c_str());
            }
            fflush(stdout);


            LOG_DBG("sampled: (%d, '%s')\n", id, token_str.c_str());

            drafts.clear();
            drafts.push_back({id});
            inp.push_back(id);
            {
                // Update context ngram cache with the newly accepted token:
                const int64_t t_start_draft_us = ggml_time_us();
                common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
                t_draft_us += ggml_time_us() - t_start_draft_us;
            }
            break;
        }

        if ((params.n_predict > 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        // KV cache management
        // clean the cache of draft tokens that weren't accepted
        if (seq_best != 0 && i_dft > 0) {
            llama_kv_cache_seq_cp(ctx, seq_best, 0, n_past-i_dft, n_past);
        }
        llama_kv_cache_seq_keep(ctx, 0);
        llama_kv_cache_seq_rm(ctx, 0, n_past, -1);

        common_batch_clear(batch_tgt);
        for (int j = 0; j < n_seq; ++j) {
            sampling_idx_store[j].clear();
        }

        // Draft already contains a single token sampled from the model:
        GGML_ASSERT(drafts.size() == 1);
        GGML_ASSERT(drafts[0].size() == 1);
        GGML_ASSERT(drafts[0][0] == inp.back());
        const int64_t t_start_draft_us = ggml_time_us();

        common_ngram_cache_draft(inp, drafts, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ngram_cache_context, ngram_cache_dynamic, ngram_cache_static);

        for (int j = 1; j < (int) drafts.size(); ++j) {
            llama_kv_cache_seq_cp(ctx, 0, j, -1, -1);
        }

        int draft_max = 0;
        for (const llama_draft_t & draft : drafts) {
            draft_max = std::max(draft_max, (int) draft.size());
        }

        if (draft_max > 1) {
            LOG("drafts:\n");
            for (const llama_draft_t & draft : drafts) {
                LOG("  - %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, draft).c_str());
            }
        }

        // FIXME wrong KV mask for converging sequences (does not seem to happen in practice).
        for (int i = 0; i < draft_max; ++i) {
            std::set<llama_token> seen_tokens;

            while (true) {
                llama_token               current_token   = -1;
                std::vector<llama_seq_id> current_seq_ids;

                for (int j = 0; j < (int) drafts.size(); ++j) {
                    if (i >= (int) drafts[j].size()) {
                        continue;
                    }

                    if (current_token == -1) {
                        if (seen_tokens.find(drafts[j][i]) != seen_tokens.end()) {
                            continue;
                        }

                        current_token = drafts[j][i];
                        seen_tokens.emplace(current_token);
                    }

                    if (drafts[j][i] != current_token) {
                        continue;
                    }

                    current_seq_ids.push_back(j);
                }

                if (current_token == -1) {
                    break;
                }

                for (const llama_seq_id & sid : current_seq_ids) {
                    sampling_idx_store[sid].push_back(batch_tgt.n_tokens);
                }
                common_batch_add(batch_tgt, current_token, n_past + i, current_seq_ids, true);
                n_drafted++;
            }
        }
        n_drafted--; // 1 out of the added token was sampled;

        t_draft_us += ggml_time_us() - t_start_draft_us;

        llama_decode(ctx, batch_tgt);
        ++n_past;
    }

    auto t_dec_end = ggml_time_us();

    // Update dynamic ngram cache with context ngram cache and save it to disk:
    common_ngram_cache_merge(ngram_cache_dynamic, ngram_cache_context);
    common_ngram_cache_save(ngram_cache_dynamic, params.lookup_cache_dynamic);

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft      = %d\n", n_draft);
    LOG_INF("n_predict    = %d\n", n_predict);
    LOG_INF("n_drafted    = %d\n", n_drafted);
    LOG_INF("t_draft_flat = %.2f ms\n", t_draft_flat_us*1e-3);
    LOG_INF("t_draft      = %.2f ms, %.2f us per token, %.2f tokens per second\n",
            t_draft_us*1e-3, 1.0f*t_draft_us/n_drafted, n_drafted/(1e-6*t_draft_us));
    LOG_INF("n_accept     = %d\n", n_accept);
    LOG_INF("accept       = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\ntarget:\n\n");
    common_perf_print(ctx, smpl);

    common_sampler_free(smpl);

    llama_batch_free(batch_tgt);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
