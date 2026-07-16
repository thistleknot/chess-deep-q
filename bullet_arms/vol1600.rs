/*
chess-deep-q goal-1600 run — jw1912/bullet STOCK recipe (examples/simple.rs), replicated
whole per the repo's replicate-before-invent law. Departures from stock (declared):
  1. data = self-generated corpus (volume-net d2 self-play, 1.34M positions,
     FEN | own-search-cp | result — purity law: no external label source)
  2. TrainingSteps scaled to the corpus: stock assumes ~100M-position superbatches;
     here batches_per_superbatch=82 makes one superbatch ~= one epoch over 1.34M.
     Stock proportions kept: 40 superbatches, LR step at 18 (drops at epochs 18/36).
Everything else — arch (768 -> 128)x2 dual-perspective SCReLU, AdamW, SCALE 400,
QA/QB quantisation, ConstantWDL 0.75, StepLR(0.001, 0.1) — is stock.
*/
use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};

const HIDDEN_SIZE: usize = 128;
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

fn main() {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    let schedule = TrainingSchedule {
        net_id: "vol1600".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 82, // ~= one epoch over the 1.34M-position corpus
            start_superbatch: 1,
            end_superbatch: 40,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.75 },
        lr_scheduler: lr::StepLR { start: 0.001, gamma: 0.1, step: 18 },
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 64,
    };

    let data_loader = loader::DirectSequentialDataLoader::new(&[
        "C:/Users/user/Documents/dev/chess-deep-q/data/corpus_vol_f_shuffled.data",
    ]);

    trainer.run(&schedule, &settings, &data_loader);
}
