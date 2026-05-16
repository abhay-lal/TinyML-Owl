import os
import numpy as np
import pandas as pd
import tensorflow as tf
from dataset import OwlSoundDataset

# --- Config & Paths ---
SAVED_MODEL_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/mbconv/buow_tinycnn_mbconv2_100_melfix_saved_model"
ROOT_OUTPUT_DIR = "/Users/ryanhoang/Desktop/OWL STUFF/TinyML-OWL-spec/TinyML-Owl/seed_experiment"

DATA_DIR = "/Users/ryanhoang/Downloads/buowset1.33"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_FILE = os.path.join(DATA_DIR, "meta", "metadata.csv")

NUM_CALIB_SAMPLES = 400
CALIB_FOLDS = [0, 1, 2]
REF_FOLDS = [0, 1, 2]


def get_input_shape(saved_model_dir):
    try:
        model = tf.saved_model.load(saved_model_dir)
        func = model.signatures["serving_default"]
        _, input_tensor = list(func.structured_input_signature[1].items())[0]
        return input_tensor.shape.as_list()
    except Exception as e:
        print(f"Can't get shape. {e}")
        return [1, 128, 241, 3]


TARGET_SHAPE = get_input_shape(SAVED_MODEL_DIR)


class ProportionalCalibGenerator:
    def __init__(self, meta_file, audio_dir, calib_folds, ref_folds, num_samples, target_shape, random_seed):
        self.meta = pd.read_csv(meta_file)
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        self.target_shape = target_shape
        self.used_filenames = []
        self.random_seed = random_seed

        reference_df = self.meta[self.meta["fold"].isin(ref_folds)]
        class_probs = reference_df["label"].value_counts(normalize=True)

        pool_df = self.meta[self.meta["fold"].isin(calib_folds)].reset_index(drop=True)

        stratified_list = []
        for label, prob in class_probs.items():
            n_target = max(1, int(round(prob * num_samples)))
            label_subset = pool_df[pool_df["label"] == label]
            n_to_take = min(len(label_subset), n_target)
            stratified_list.append(label_subset.sample(n=n_to_take, random_state=self.random_seed))

        self.final_calib_df = (
            pd.concat(stratified_list)
            .sample(frac=1, random_state=self.random_seed)
            .reset_index(drop=True)
        )
        self.ds = OwlSoundDataset(self.final_calib_df, self.audio_dir)

    def __call__(self):
        self.used_filenames = []
        taken = 0

        for idx in range(len(self.final_calib_df)):
            try:
                x, _ = self.ds[idx]
                x_np = np.expand_dims(x.detach().cpu().numpy().astype(np.float32), axis=0)

                if x_np.shape[1] == 1 and self.target_shape[1] == 3:
                    x_np = np.repeat(x_np, 3, axis=1)

                if list(x_np.shape) != self.target_shape:
                    continue

                fname = self.final_calib_df.iloc[idx]["segment_path"]
                self.used_filenames.append(fname)

                yield [x_np]

                taken += 1
                if taken >= self.num_samples:
                    break
            except Exception as e:
                print(f"Error at index {idx}: {e}")
                continue


def run_conversion(mode, seed, output_dir):
    out_path = os.path.join(output_dir, f"buow_model_{mode}_proportional.tflite")
    log_path = os.path.join(output_dir, f"calib_log_proportional_{mode}.csv")

    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    gen = ProportionalCalibGenerator(
        META_FILE, AUDIO_DIR, CALIB_FOLDS, REF_FOLDS, NUM_CALIB_SAMPLES, TARGET_SHAPE, random_seed=seed
    )

    print(f"\n--- Processing: {mode} (Proportional) | Seed: {seed} ---")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = gen

    if mode == "int8":
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    elif mode == "int16":
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]

    try:
        tflite_model = converter.convert()
        with open(out_path, "wb") as f:
            f.write(tflite_model)

        pd.DataFrame(gen.used_filenames, columns=["segment_path"]).to_csv(log_path, index=False)
        print(f"Success! Seed {seed} | Samples logged: {len(gen.used_filenames)}")
    except Exception as e:
        print(f"Error on Seed {seed}: {e}")


if __name__ == "__main__":
    for seed in range(1, 21):
        seed_dir = os.path.join(ROOT_OUTPUT_DIR, str(seed))
        os.makedirs(seed_dir, exist_ok=True)

        for m in ["int8", "int16"]:
            run_conversion(m, seed, seed_dir)