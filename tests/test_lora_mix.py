"""The LoRA mixer: two LoRAs into one new file, exactly (docs/26 §10).

Asked for: "mix them into a single new lora that i can put into a choose
directory in the lora directory. I should be forced to choose output folder
everytime. I can also choose to name it something different. Once the lora is
created, our extension should remember it was made by the extension, i would
like a placed holder image until i set one myself."

The mix is checked against Forge's own arithmetic, written out here apart from
the mixer: Forge applies a layer as ``strength * alpha / rank * up @ down``
(its vendored ComfyUI ``weight_adapter/lora.py``, the Tucker middle folded as
there), so the mix of two LoRAs at ``s_a`` and ``s_b`` must change every layer
by ``s_a * delta_a + s_b * delta_b`` -- read off the written file.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import mc_lora_jobs as jobs
import mc_lora_mix as mix
import mc_lora_render

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------- #
# LoRA files, and Forge's arithmetic
# --------------------------------------------------------------------------- #


def write_lora(path: Path, tensors: dict, metadata: dict | None = None) -> Path:
    """``tensors``: name -> (array, dtype)."""
    out = [mix.OutTensor(name, dtype, np.asarray(values).shape, mix.encode(np.asarray(values, np.float32), dtype))
           for name, (values, dtype) in tensors.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    mix.write_safetensors(path, metadata or {}, out)
    return path


def pair(prefix: str, out: int, inp: int, rank: int, *, alpha: float | None, dtype="F32", style="kohya",
         kernel=None):
    down_shape = (rank, inp, *kernel) if kernel else (rank, inp)
    up_shape = (out, rank, 1, 1) if kernel else (out, rank)
    down = RNG.standard_normal(down_shape).astype(np.float32) * 0.1
    up = RNG.standard_normal(up_shape).astype(np.float32) * 0.1
    names = {"kohya": (".lora_down.weight", ".lora_up.weight"), "peft": (".lora_A.weight", ".lora_B.weight"),
             "mochi": (".lora_A", ".lora_B")}[style]
    tensors = {prefix + names[0]: (down, dtype), prefix + names[1]: (up, dtype)}
    if alpha is not None:
        tensors[prefix + ".alpha"] = (np.array(alpha, np.float32), "F32")
    return tensors


def forge_deltas(path: Path) -> dict:
    """Each layer's change at strength 1, by Forge's formula, keyed by the
    layer's canonical name. Written apart from the mixer's own reading."""
    _meta, tensors = mix.read_safetensors(path)
    arrays = {name: tensor.floats() for name, tensor in tensors.items()}
    out = {}
    ups = [(name[: -len(suffix)], suffix) for name in arrays for suffix in
           (".lora_up.weight", ".lora_B.weight", ".lora_B") if name.endswith(suffix)]
    downs = {".lora_up.weight": ".lora_down.weight", ".lora_B.weight": ".lora_A.weight", ".lora_B": ".lora_A"}
    for prefix, suffix in ups:
        mat1 = arrays[prefix + suffix]
        mat2 = arrays[prefix + downs[suffix]]
        alpha = arrays.get(prefix + ".alpha")
        scale = float(alpha) / mat2.shape[0] if alpha is not None else 1.0
        if prefix + ".lora_mid.weight" in arrays:
            mat3 = arrays[prefix + ".lora_mid.weight"]
            final = [mat2.shape[1], mat2.shape[0], mat3.shape[2], mat3.shape[3]]
            mat2 = (np.transpose(mat2, (1, 0, 2, 3)).reshape(mat2.shape[1], -1)
                    @ np.transpose(mat3, (1, 0, 2, 3)).reshape(mat3.shape[1], -1))
            mat2 = np.transpose(mat2.reshape(final), (1, 0, 2, 3))
        out[mix.canonical(prefix)] = scale * (mat1.reshape(mat1.shape[0], -1) @ mat2.reshape(mat2.shape[0], -1))
    for name, values in arrays.items():
        for role in mix.DIFF_ROLES:
            if name.endswith("." + role):
                out[mix.canonical(name[: -len(role) - 1]) + "#" + role] = values
    return out


def expected(first: dict, second: dict, s_a: float, s_b: float) -> dict:
    out = {}
    for key in set(first) | set(second):
        total = 0
        if key in first:
            total = total + s_a * first[key]
        if key in second:
            total = total + s_b * second[key]
        out[key] = total
    return out


def mix_files(tmp_path, a: Path, b: Path, s_a: float, s_b: float, name="mixed") -> Path:
    meta_a, tensors_a = mix.read_safetensors(a)
    meta_b, tensors_b = mix.read_safetensors(b)
    outcome = mix.mix(mix.Source("A", tensors_a, meta_a, s_a), mix.Source("B", tensors_b, meta_b, s_b))
    target = tmp_path / f"{name}.safetensors"
    mix.write_safetensors(target, {"ss_output_name": name}, outcome.tensors)
    return target


def assert_deltas(found: dict, wanted: dict, *, tolerance=1e-5):
    assert set(found) == set(wanted), (sorted(found), sorted(wanted))
    for key in wanted:
        scale = max(1e-6, float(np.abs(wanted[key]).max()))
        assert float(np.abs(found[key] - wanted[key]).max()) / scale < tolerance, key


# --------------------------------------------------------------------------- #
# The mix is exact
# --------------------------------------------------------------------------- #


class TestTheMix:
    def test_every_layer_changes_by_the_two_at_their_strengths(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", {
            **pair("lora_unet_layers_0_attn_qkv", 24, 16, 4, alpha=2.0),
            **pair("lora_unet_layers_0_mlp_up", 32, 16, 4, alpha=4.0),
        })
        b = write_lora(tmp_path / "b.safetensors", {
            **pair("lora_unet_layers_0_attn_qkv", 24, 16, 8, alpha=1.0),
            **pair("lora_unet_layers_1_attn_out", 16, 16, 8, alpha=8.0),
        })
        out = mix_files(tmp_path, a, b, 0.8, -0.5)

        assert_deltas(forge_deltas(out), expected(forge_deltas(a), forge_deltas(b), 0.8, -0.5))

    def test_a_layer_both_change_is_one_layer_of_the_two_ranks(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_x", 12, 10, 4, alpha=4.0))
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x", 12, 10, 6, alpha=3.0))
        _meta, tensors = mix.read_safetensors(mix_files(tmp_path, a, b, 1.0, 1.0))

        assert set(tensors) == {"lora_unet_x.lora_down.weight", "lora_unet_x.lora_up.weight", "lora_unet_x.alpha"}
        assert tensors["lora_unet_x.lora_down.weight"].shape == (10, 10)
        assert tensors["lora_unet_x.lora_up.weight"].shape == (12, 10)

    def test_the_stronger_lora_is_written_as_it_was(self, tmp_path):
        """Its up matrix is copied, not scaled: the larger scale goes into the
        alpha, and a half-precision LoRA loses nothing to the mix."""
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_x", 12, 10, 4, alpha=4.0, dtype="F16"))
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x", 12, 10, 4, alpha=4.0, dtype="F16"))
        _m, original = mix.read_safetensors(a)
        _m, tensors = mix.read_safetensors(mix_files(tmp_path, a, b, 1.0, 0.25))

        up = tensors["lora_unet_x.lora_up.weight"]
        assert up.dtype == "F16"
        assert np.array_equal(up.floats()[:, :4], original["lora_unet_x.lora_up.weight"].floats())
        assert np.array_equal(tensors["lora_unet_x.lora_down.weight"].floats()[:4],
                              original["lora_unet_x.lora_down.weight"].floats())

    def test_a_negative_strength_flips_the_up_matrix_not_the_alpha(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_x", 8, 8, 2, alpha=2.0))
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_y", 8, 8, 2, alpha=2.0))
        out = mix_files(tmp_path, a, b, -1.5, 1.0)
        _m, tensors = mix.read_safetensors(out)

        assert tensors["lora_unet_x.alpha"].scalar() > 0
        assert_deltas(forge_deltas(out), expected(forge_deltas(a), forge_deltas(b), -1.5, 1.0))

    def test_a_layer_in_one_lora_at_strength_zero_is_left_out(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_x", 8, 8, 2, alpha=2.0))
        b = write_lora(tmp_path / "b.safetensors", {**pair("lora_unet_x", 8, 8, 2, alpha=2.0),
                                                     **pair("lora_unet_y", 8, 8, 2, alpha=2.0),
                                                     "lora_unet_z.diff_b": (np.ones(8, np.float32), "F32")})
        _m, tensors = mix.read_safetensors(mix_files(tmp_path, a, b, 1.0, 0.0))
        assert not any(name.startswith(("lora_unet_y", "lora_unet_z")) for name in tensors), sorted(tensors)

    def test_kohya_and_peft_spellings_of_one_layer_are_one_layer(self, tmp_path):
        """Forge maps every spelling of a layer to one weight, and two names for
        it in one file overwrite each other there: written once, in the first
        LoRA's spelling."""
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_layers_0_attn_qkv", 12, 10, 4, alpha=4.0))
        b = write_lora(tmp_path / "b.safetensors", {
            **pair("diffusion_model.layers.0.attn.qkv", 12, 10, 4, alpha=None, style="peft"),
            **pair("layers.1.mlp.up", 12, 10, 4, alpha=None, style="mochi"),
        })
        out = mix_files(tmp_path, a, b, 1.0, 0.7)
        _m, tensors = mix.read_safetensors(out)

        assert "lora_unet_layers_0_attn_qkv.lora_up.weight" in tensors
        assert not any(name.startswith("diffusion_model.layers.0") for name in tensors)
        assert "layers.1.mlp.up.lora_up.weight" in tensors
        assert_deltas(forge_deltas(out), expected(forge_deltas(a), forge_deltas(b), 1.0, 0.7))

    def test_half_beside_bfloat16_is_written_as_bfloat16(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_x", 12, 10, 4, alpha=4.0, dtype="F16"))
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x", 12, 10, 4, alpha=2.0, dtype="BF16"))
        out = mix_files(tmp_path, a, b, 1.0, 1.0)
        _m, tensors = mix.read_safetensors(out)

        assert tensors["lora_unet_x.lora_up.weight"].dtype == "BF16"
        assert_deltas(forge_deltas(out), expected(forge_deltas(a), forge_deltas(b), 1.0, 1.0), tolerance=2e-2)

    def test_bfloat16_rounds_to_nearest_even(self):
        values = np.array([1.0, 1.00390625, 1.01171875, -2.5, 0.0], dtype=np.float32)
        written = np.frombuffer(mix.encode(values, "BF16"), dtype="<u2")
        assert written.tolist() == [0x3F80, 0x3F80, 0x3F82, 0xC020, 0x0000]
        back = mix.Tensor("BF16", (5,), memoryview(written.tobytes())).floats()
        assert back.tolist() == [1.0, 1.0, 1.015625, -2.5, 0.0]

    def test_a_tucker_middle_is_folded_as_forge_folds_it(self, tmp_path):
        tucker = pair("lora_unet_conv", 6, 4, 3, alpha=3.0, kernel=(1, 1))
        tucker["lora_unet_conv.lora_mid.weight"] = (RNG.standard_normal((3, 3, 3, 3)).astype(np.float32) * 0.1, "F32")
        a = write_lora(tmp_path / "a.safetensors", tucker)
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_conv", 6, 4, 2, alpha=2.0, kernel=(3, 3)))
        out = mix_files(tmp_path, a, b, 0.6, 1.0)

        assert_deltas(forge_deltas(out), expected(forge_deltas(a), forge_deltas(b), 0.6, 1.0))

    def test_diff_patches_are_summed(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", {"diffusion_model.norm.diff_b": (np.ones(5, np.float32), "F32"),
                                                     **pair("diffusion_model.attn", 6, 6, 2, alpha=2.0)})
        b = write_lora(tmp_path / "b.safetensors", {"diffusion_model.norm.diff_b": (np.full(5, 2.0, np.float32), "F32"),
                                                     **pair("diffusion_model.attn", 6, 6, 2, alpha=2.0)})
        _m, tensors = mix.read_safetensors(mix_files(tmp_path, a, b, 0.5, 0.25))
        assert tensors["diffusion_model.norm.diff_b"].floats().tolist() == [1.0] * 5

    def test_layers_of_one_scheme_that_do_not_overlap_are_both_kept(self, tmp_path):
        """One LoRA trained on attention, the other on the MLP: no shared layer,
        and the same spelling, so no layer can be reached twice."""
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_attn", 8, 8, 2, alpha=2.0))
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_mlp", 8, 8, 2, alpha=2.0))
        out = mix_files(tmp_path, a, b, 1.0, 1.0)
        assert set(forge_deltas(out)) == {"attn", "mlp"}


class TestRefusals:
    @pytest.mark.parametrize("mark, kind", [("lora_unet_x.dora_scale", "DoRA"), ("lora_unet_x.hada_w1_a", "LoHa"),
                                             ("lora_unet_x.lokr_w1", "LoKr")])
    def test_what_cannot_be_mixed_exactly_is_named(self, tmp_path, mark, kind):
        a = write_lora(tmp_path / "a.safetensors", {**pair("lora_unet_x", 8, 8, 2, alpha=2.0),
                                                     mark: (np.ones((8, 1), np.float32), "F32")})
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x", 8, 8, 2, alpha=2.0))
        with pytest.raises(mix.MixError, match=kind):
            mix_files(tmp_path, a, b, 1.0, 1.0)

    def test_a_tensor_of_an_unknown_kind_is_refused_not_dropped(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", {**pair("lora_unet_x", 8, 8, 2, alpha=2.0),
                                                     "lora_unet_x.lora_B.bias": (np.ones(8, np.float32), "F32")})
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x", 8, 8, 2, alpha=2.0))
        with pytest.raises(mix.MixError, match="does not read"):
            mix_files(tmp_path, a, b, 1.0, 1.0)

    def test_a_layer_of_another_size_means_another_model(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_x", 8, 8, 2, alpha=2.0))
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x", 8, 16, 2, alpha=2.0))
        with pytest.raises(mix.MixError, match="different size"):
            mix_files(tmp_path, a, b, 1.0, 1.0)

    def test_no_shared_layer_and_different_spellings_is_refused(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", pair("lora_unet_double_blocks_0_qkv", 8, 8, 2, alpha=2.0))
        b = write_lora(tmp_path / "b.safetensors",
                       pair("transformer.transformer_blocks.0.attn.to_q", 8, 8, 2, alpha=None, style="peft"))
        with pytest.raises(mix.MixError, match="share no layer"):
            mix_files(tmp_path, a, b, 1.0, 1.0)

    def test_a_layer_named_twice_in_one_lora_is_refused(self, tmp_path):
        a = write_lora(tmp_path / "a.safetensors", {**pair("lora_unet_x_y", 8, 8, 2, alpha=2.0),
                                                     **pair("diffusion_model.x.y", 8, 8, 2, alpha=None, style="peft")})
        b = write_lora(tmp_path / "b.safetensors", pair("lora_unet_x_y", 8, 8, 2, alpha=2.0))
        with pytest.raises(mix.MixError, match="twice"):
            mix_files(tmp_path, a, b, 1.0, 1.0)

    def test_a_file_that_is_not_safetensors(self, tmp_path):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"not a model")
        with pytest.raises(mix.MixError, match="not a safetensors"):
            mix.read_safetensors(bad)


# --------------------------------------------------------------------------- #
# Checking a request: two LoRAs, a folder every time, a name
# --------------------------------------------------------------------------- #


class FakeNetworks:
    """Forge's ``networks`` module, as much as the mixer reads."""

    def __init__(self, files: dict):
        self.available_networks = {name: SimpleNamespace(filename=str(path), alias=name)
                                   for name, path in files.items()}
        self.available_network_aliases = dict(self.available_networks)
        self.updated = []

    def update_available_networks_by_names(self, names):
        self.updated.extend(names)

    def list_available_networks(self):
        self.updated.append("*")


@pytest.fixture
def world(tmp_path, lora_root, monkeypatch):
    """A LoRA folder with two LoRAs in a subfolder, Forge knowing both."""
    root = tmp_path / "Lora"
    (root / "Krea2").mkdir(parents=True)
    (root / "Krea2" / "LIGHTING").mkdir()
    (root / ".hidden").mkdir()
    (root / "out").mkdir()
    a = write_lora(root / "Krea2" / "alpha.safetensors", pair("lora_unet_x", 8, 8, 2, alpha=2.0),
                   {"ss_output_name": "alpha", "sshs_model_hash": "deadbeef", "modelspec.architecture": "krea-2/lora"})
    b = write_lora(root / "Krea2" / "beta.safetensors", pair("lora_unet_x", 8, 8, 4, alpha=4.0),
                   {"ss_output_name": "beta", "modelspec.architecture": "krea-2/lora"})
    (root / "Krea2" / "alpha.json").write_text(json.dumps({"activation text": "red hat, smile", "sd version": "Krea2"}))
    (root / "Krea2" / "beta.json").write_text(json.dumps({"activation text": "smile, blue coat", "sd version": "Krea2"}))
    networks = FakeNetworks({"alpha": a, "beta": b})
    monkeypatch.setattr(mix, "forge_networks", lambda: networks)
    monkeypatch.setattr(mix, "lora_directories", lambda: [str(root)])
    return SimpleNamespace(root=root, a=a, b=b, networks=networks)


class TestTheRequest:
    def test_the_folders_are_the_lora_folder_and_every_folder_under_it(self, world):
        labels = [label for label, _ in mix.folders()]
        assert labels == ["LoRA folder", "Krea2", "Krea2/LIGHTING", "out"]

    def test_with_several_lora_folders_each_is_named(self, world, tmp_path):
        other = tmp_path / "More"
        (other / "SDXL").mkdir(parents=True)
        labels = [label for label, _ in mix.folders([str(world.root), str(other)])]
        assert labels == ["Lora", "Lora/Krea2", "Lora/Krea2/LIGHTING", "Lora/out", "More", "More/SDXL"]

    def test_the_pickers_list_each_lora_by_its_place(self, world):
        assert mix.loras() == [("Krea2/alpha", "alpha"), ("Krea2/beta", "beta")]

    def test_a_good_request(self, world):
        request = mix.check("alpha", 1, "beta", 0.5, "out", "")
        assert request.folder == world.root / "out" and request.name == "mix_alpha_beta"
        assert request.target == world.root / "out" / "mix_alpha_beta.safetensors"

    @pytest.mark.parametrize("args, reason", [
        (("alpha", 1, "", 1, "out", ""), "choose two LoRAs"),
        (("alpha", 1, "alpha", 1, "out", ""), "two different"),
        (("alpha", 1, "beta", 1, "", ""), "choose the folder"),
        (("alpha", 1, "beta", 1, "../elsewhere", ""), "not one of the LoRA folders"),
        (("alpha", 0, "beta", 0, "out", ""), "both strengths are 0"),
        (("alpha", 9, "beta", 1, "out", ""), "outside"),
        (("alpha", 1, "beta", 1, "out", "a/b"), "cannot hold"),
        (("alpha", 1, "beta", 1, "out", "CON"), "cannot be a file"),
        (("alpha", 1, "beta", 1, "out", "beta"), "Forge already has"),
        (("ghost", 1, "beta", 1, "out", ""), "not a LoRA Forge knows"),
    ])
    def test_refused_before_anything_is_read(self, world, args, reason):
        with pytest.raises(mix.MixError, match=reason):
            mix.check(*args)

    def test_a_name_in_the_folder_is_refused_whatever_its_case(self, world):
        (world.root / "out" / "Mine.safetensors").write_bytes(b"x")
        with pytest.raises(mix.MixError, match="already in that folder"):
            mix.check("alpha", 1, "beta", 1, "out", "mine")

    def test_the_finished_file_never_replaces_one(self, tmp_path):
        temp = tmp_path / "t.part"
        temp.write_bytes(b"new")
        target = tmp_path / "x.safetensors"
        target.write_bytes(b"old")
        with pytest.raises(mix.MixError, match="appeared"):
            mix._link_into_place(temp, target)
        assert target.read_bytes() == b"old"


# --------------------------------------------------------------------------- #
# Making one: the file, what is remembered, the placeholder
# --------------------------------------------------------------------------- #


class TestMakingOne:
    def test_the_file_its_metadata_and_what_is_remembered(self, world):
        entry = mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "My Mix"))
        path = world.root / "out" / "My Mix.safetensors"

        meta, _tensors = mix.read_safetensors(path)
        assert meta["ss_output_name"] == "My Mix", "Forge reads the alias from it"
        assert "sshs_model_hash" not in meta, "the first LoRA's hash would have been taken for the mix's"
        assert meta["modelspec.architecture"] == "krea-2/lora"
        record = mix.mix_record(meta)
        assert record["made_by"] == mix.MADE_BY
        assert [(item["name"], item["strength"]) for item in record["inputs"]] == [("alpha", 1.0), ("beta", 0.5)]
        assert all(len(item["sha256"]) == 64 for item in record["inputs"])

        remembered = mix.load_registry()
        assert len(remembered) == 1 and remembered[0]["name"] == "My Mix" and remembered[0]["folder"] == "out"
        assert remembered[0]["sha256"] == entry["sha256"]
        assert world.networks.updated == ["My Mix"], "Forge's list learns the new LoRA"

    def test_a_placeholder_until_you_set_your_own(self, world):
        mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "pic"))
        placeholder = world.root / "out" / "pic.png"
        assert placeholder.is_file()
        from PIL import Image

        with Image.open(placeholder) as image:
            assert image.size == mix.PLACEHOLDER_SIZE
        assert mix.mixes()[0]["picture"] == "placeholder"

        # Forge's Replace preview writes the same file: the picture is yours.
        Image.new("RGB", (64, 96), (1, 2, 3)).save(placeholder)
        assert mix.mixes()[0]["picture"] == "yours"

    def test_a_picture_of_yours_beside_it_takes_the_placeholders_place(self, world):
        mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "pic"))
        (world.root / "out" / "pic.preview.webp").write_bytes(b"RIFF....WEBP")
        assert mix.sweep_placeholders() == 1
        assert not (world.root / "out" / "pic.png").exists()
        assert mix.mixes()[0]["picture"] == "yours"
        assert mix.sweep_placeholders() == 0

    def test_the_placeholder_is_written_in_forges_images_format(self, world):
        from modules import shared

        shared.opts.samples_format = "jpg"
        try:
            mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "pic"))
        finally:
            del shared.opts.samples_format
        assert (world.root / "out" / "pic.jpg").is_file() and not (world.root / "out" / "pic.png").exists()

    def test_forges_user_metadata_says_it_is_a_mix(self, world):
        mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "pic"))
        meta = json.loads((world.root / "out" / "pic.json").read_text(encoding="utf-8"))
        assert meta["description"] == "Mixed by Model Chain from alpha at 1 and beta at 0.5."
        assert meta["activation text"] == "red hat, smile, blue coat"
        assert meta["sd version"] == "Krea2"

    def test_the_mix_changes_the_model_as_the_two_did(self, world):
        mix.run(mix.check("alpha", 0.75, "beta", -0.5, "out", "check"))
        assert_deltas(forge_deltas(world.root / "out" / "check.safetensors"),
                      expected(forge_deltas(world.a), forge_deltas(world.b), 0.75, -0.5))


class TestTheJob:
    def wait(self, job):
        deadline = time.time() + 20
        while jobs.MANAGER.get(job.id).snapshot().active and time.time() < deadline:
            time.sleep(0.02)
        return jobs.MANAGER.get(job.id).snapshot()

    def test_a_mix_runs_as_a_job_and_logs_no_name(self, world, caplog):
        with caplog.at_level(logging.INFO, logger="model_chain"):
            snapshot = self.wait(mix.submit("alpha", 1, "beta", 1, "out", "Secret Name"))
        assert snapshot.state == jobs.DONE and snapshot.summary.startswith("Saved Secret Name in out"), snapshot
        lines = [record.getMessage() for record in caplog.records if record.name == "model_chain"]
        assert any("made a LoRA mix" in line for line in lines)
        assert not any("Secret Name" in line or "alpha" in line or str(world.root) in line for line in lines), lines

    def test_a_refusal_in_the_job_is_its_answer_not_a_failure(self, world, monkeypatch):
        request = mix.check("alpha", 1, "beta", 1, "out", "late")
        (world.root / "out" / "late.safetensors").write_bytes(b"made meanwhile")
        monkeypatch.setattr(mix, "check", lambda *args, **kwargs: request)
        snapshot = self.wait(mix.submit("alpha", 1, "beta", 1, "out", "late"))
        assert snapshot.state == jobs.DONE and snapshot.summary.startswith("Not made:"), snapshot
        assert (world.root / "out" / "late.safetensors").read_bytes() == b"made meanwhile"


# --------------------------------------------------------------------------- #
# The page and the dialog
# --------------------------------------------------------------------------- #


class TestThePage:
    def test_the_folder_is_emptied_once_a_mix_is_queued_and_kept_on_a_refusal(self, world):
        import mc_lora_mix_ui as ui

        refused = ui.do_mix("alpha", 1, "beta", 1, "", "")
        assert "choose the folder" in refused[0] and refused[1] == {} and refused[3] is False

        status, folder, job_id, poll = ui.do_mix("alpha", 1, "beta", 1, "out", "")
        assert folder == {"value": None} and poll is True and job_id > 0, "chosen again for the next mix"
        deadline = time.time() + 20
        while not ui.job_answer(job_id)[1] and time.time() < deadline:
            time.sleep(0.02)
        text, done = ui.job_answer(job_id)
        assert done and "Saved mix_alpha_beta" in text

    def test_the_name_box_shows_the_default_once_both_are_chosen(self, world):
        import mc_lora_mix_ui as ui

        assert ui.suggest_name("alpha", "beta") == {"placeholder": "mix_alpha_beta"}
        assert "when left empty" in ui.suggest_name("alpha", None)["placeholder"]

    def test_the_tab_has_a_mix_page_whose_folder_starts_empty(self, world, host, monkeypatch):
        import gradio as gr

        import mc_lora_tab

        made = []
        original = gr.Dropdown

        class Dropdown(original):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                made.append(kwargs)

        monkeypatch.setattr(gr, "Dropdown", Dropdown)
        mc_lora_tab.create_tab()
        by_id = {kwargs.get("elem_id"): kwargs for kwargs in made}
        assert by_id["mc-lora-mix-folder"]["value"] is None
        assert by_id["mc-lora-mix-folder"]["choices"] == ["LoRA folder", "Krea2", "Krea2/LIGHTING", "out"]
        assert by_id["mc-lora-mix-first"]["filterable"] is True
        assert by_id["mc-lora-mix-first"]["choices"] == [("Krea2/alpha", "alpha"), ("Krea2/beta", "beta")]

    def test_the_list_says_what_each_mix_was_made_from(self, world):
        mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "listed"))
        html = mc_lora_render.mixes_html(mix.mixes())
        assert "listed" in html and "alpha ×1 + beta ×0.5" in html and "Placeholder picture" in html
        assert "No mixes made here yet" in mc_lora_render.mixes_html([])

    def test_the_dialog_knows_a_mix_and_offers_no_civitai_fetch(self, world, host):
        import mc_lora_details

        mix.run(mix.check("alpha", 1, "beta", 0.5, "out", "seen"))
        path = world.root / "out" / "seen.safetensors"
        meta, _ = mix.read_safetensors(path)
        data = mc_lora_details.gather(None, "seen", item={"filename": str(path), "metadata": meta},
                                      hash_files=False)
        assert data["mix"]["inputs"][0]["name"] == "alpha"
        assert data["fetch"] == {}
        assert "Mixed from" in mc_lora_render.mix_html(data["mix"])

        plain = mc_lora_details.gather(None, "alpha", item={"filename": str(world.a), "metadata": {}},
                                       hash_files=False)
        assert plain["mix"] is None and plain["fetch"], "a LoRA from elsewhere is still offered a fetch"
