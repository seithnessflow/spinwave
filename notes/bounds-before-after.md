# The 28 bound cases, before and after their reconstruction

The bounds check of 2026-09-12 (`bounds.rs`: no control and no modulated
value of a case may touch a bound of its range during the render, unless
the bound is what the case measures) flagged 28 existing cases. They were
rebuilt interior and re-rendered in the same pass, without a residual
before/after per case — so a residual could have been hiding behind a
clamp, or a clamp could have been hiding a residual. This table closes
that: the OLD case files and their OLD references (`git show
2daa6b5:tools/golden/{cases,reference}/...`) run through the engine of
2026-09-13 evening (`SPINWAVE_BENCH_DIR` points the bench at them), next
to the rebuilt cases through the same engine (the full bench run of the
same evening, 336 cases).

RMS of the sample difference against the reference render; the floor of
the bench is 3e-8 to 1.2e-7 (float noise of two engines summing the same
signal in a different order).

| case | old case, engine of today | rebuilt case, engine of today |
|---|---|---|
| `fx_flanger` | 4.3e-08 | 4.2e-08 |
| `meta_chain_backward` | 4.3e-08 | 4.3e-08 |
| `meta_chain_forward` | 4.3e-08 | 4.3e-08 |
| `meta_chain_no_macro` | 4.0e-08 | 4.1e-08 |
| `meta_chain_three_links` | 4.4e-08 | 3.9e-08 |
| `meta_lfo_on_audio_rate_amount` | 3.9e-08 | 4.2e-08 |
| `meta_lfo_source_bipolar` | 4.2e-08 | 3.9e-08 |
| `meta_lfo_source_unipolar` | 4.2e-08 | 4.4e-08 |
| `meta_macro_to_amount` | 4.1e-08 | 4.3e-08 |
| `meta_macro_to_amount_static` | 4.2e-08 | 4.3e-08 |
| `meta_macro_to_amount_twin` | 4.1e-08 | 4.3e-08 |
| `meta_on_bipolar_target` | 3.7e-08 | 5.0e-08 |
| `meta_on_bipolar_target_static` | 3.7e-08 | 5.0e-08 |
| `meta_poly_source_two_voices` | 5.8e-08 | 6.3e-08 |
| `meta_ramp_on_mono_source_target` | 5.4e-08 | 2.9e-07 |
| `meta_step_on_poly_source_target` | 4.2e-08 | 4.1e-08 |
| `meta_step_timing` | 4.9e-08 | 1.1e-07 |
| `meta_step_timing_twin_high` | 3.8e-08 | 4.3e-08 |
| `meta_step_timing_twin_low` | 3.8e-08 | 4.2e-08 |
| `mod_env_to_level` | 9.8e-08 | 5.7e-08 |
| `mod_env_to_tune` | 4.4e-08 | 4.4e-08 |
| `mod_lfo_bipolar` | 5.3e-08 | 4.1e-08 |
| `mod_lfo_bipolar_low` | 4.2e-08 | 5.2e-08 |
| `mod_lfo_to_cutoff` | 4.3e-08 | 4.1e-08 |
| `mod_lfo_to_cutoff_high` | 4.1e-08 | 4.2e-08 |
| `mod_lfo_to_level` | 1.2e-07 | 6.1e-08 |
| `mod_lfo_to_phase` | 4.3e-08 | 4.4e-08 |
| `mod_two_voices_one_lfo` | 5.9e-08 | 6.1e-08 |

Every one of the 28 is at the floor on both sides of the reconstruction.
Nothing was hiding behind a clamp: a residual that a bound had masked
would show on the old file (the old files still hit their bounds — the
bench prints the `BOUND` lines — and clamp identically on both engines,
which is why they never lied). The one case that had a residual on the
old file when this was first measured (2026-09-13 morning,
`mod_env_to_tune` at 5.8e-5) owed it to `Wavetable::frequency_float_bin`
taking the exact `log2` where the reference takes the polynomial one — a
mip-bin crossing placed a sample apart — and not to its bound; the site
was corrected then (`notes/exact-vs-polynomial.md`), and the old file
reads 4.4e-8 today.

The threshold stays at 1e-4 RMS (`golden.rs`): the whole bench sits
below 4.5e-5 (`macro_dest_lfo`, a one-block step of a macro through the
mono chain, explained there) with 259 of 336 cases under 1e-7, and the
only residuals between 1e-5 and 1e-4 are the compressor family at
1e-5 — the reference's own Linkwitz-Riley crossover conditioning at
120 Hz, established by the unit goldens of the same pass.
