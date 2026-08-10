import json
import os
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "app", "models")

GENDER_IN = os.path.join(MODEL_DIR, "crayfish_gender_resnet50.keras")
GROWTH_IN = os.path.join(MODEL_DIR, "crayfish_growth_resnet50.keras")

GENDER_OUT = os.path.join(MODEL_DIR, "crayfish_gender_resnet50_patched.keras")
GROWTH_OUT = os.path.join(MODEL_DIR, "crayfish_growth_resnet50_patched.keras")


def normalize_dtype(value):
    # Turn serialized dtype policy objects into plain dtype strings like "float32"
    if isinstance(value, dict):
        class_name = value.get("class_name")
        config = value.get("config", {})

        if class_name == "DTypePolicy":
            name = config.get("name")
            if isinstance(name, str):
                return name

        # recursively normalize dict contents too
        return {k: normalize_dtype(v) for k, v in value.items()}

    if isinstance(value, list):
        return [normalize_dtype(v) for v in value]

    return value


def patch_obj(obj):
    if isinstance(obj, dict):
        class_name = obj.get("class_name")
        config = obj.get("config")

        # normalize any dtype entry anywhere in the config tree
        if isinstance(config, dict) and "dtype" in config:
            config["dtype"] = normalize_dtype(config["dtype"])

        if class_name == "InputLayer" and isinstance(config, dict):
            # remove legacy keys that break current loader
            config.pop("optional", None)

            # convert old key
            if "batch_shape" in config:
                config["batch_input_shape"] = config.pop("batch_shape")

            # Keras rejects both at the same time
            if "batch_input_shape" in config and "input_shape" in config:
                config.pop("input_shape", None)

        for v in obj.values():
            patch_obj(v)

    elif isinstance(obj, list):
        for item in obj:
            patch_obj(item)


def patch_keras_file(input_path, output_path):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Missing file: {input_path}")

    with zipfile.ZipFile(input_path, "r") as zin:
        names = zin.namelist()

        if "config.json" not in names:
            raise ValueError(f"{input_path} has no config.json")

        config_data = zin.read("config.json").decode("utf-8")
        config_json = json.loads(config_data)

        patch_obj(config_json)

        patched_config = json.dumps(config_json, ensure_ascii=False)

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                if name == "config.json":
                    zout.writestr("config.json", patched_config)
                else:
                    zout.writestr(name, zin.read(name))

    print(f"Patched: {input_path}")
    print(f"Saved as: {output_path}")


def main():
    print("MODEL_DIR:", MODEL_DIR)
    print("Files:", os.listdir(MODEL_DIR))

    patch_keras_file(GENDER_IN, GENDER_OUT)
    patch_keras_file(GROWTH_IN, GROWTH_OUT)

    print("Done patching both models.")


if __name__ == "__main__":
    main()