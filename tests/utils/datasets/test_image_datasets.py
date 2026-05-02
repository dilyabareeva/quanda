import datasets
import pytest
import torch

from quanda.utils.datasets.image_datasets import HFtoTV


def _make_hf_image_dataset(image_key="image", num=3):
    return datasets.Dataset.from_dict(
        {
            image_key: [[[float(i)]] for i in range(num)],
            "label": list(range(num)),
        }
    )


@pytest.mark.utils
def test_hftotv_get_label_returns_int():
    ds = _make_hf_image_dataset()
    wrapper = HFtoTV(ds)

    assert wrapper.get_label(0) == 0
    assert wrapper.get_label(2) == 2


@pytest.mark.utils
def test_hftotv_get_label_caches_label_column():
    """The first get_label call should pull the whole label column once;
    subsequent calls reuse the cached list (avoids re-decoding images)."""
    ds = _make_hf_image_dataset()
    wrapper = HFtoTV(ds)

    assert wrapper._labels_cache is None
    wrapper.get_label(0)
    assert wrapper._labels_cache is not None
    cached = wrapper._labels_cache
    wrapper.get_label(1)
    assert wrapper._labels_cache is cached


@pytest.mark.utils
def test_hftotv_get_label_accepts_tensor_index():
    ds = _make_hf_image_dataset()
    wrapper = HFtoTV(ds)
    assert wrapper.get_label(torch.tensor(1)) == 1


@pytest.mark.utils
def test_hftotv_get_label_uses_label_override():
    """When label_override is set, get_label returns it without touching
    the dataset — the column-cache must stay unset."""
    ds = _make_hf_image_dataset()
    wrapper = HFtoTV(ds, label_override=42)

    assert wrapper.get_label(0) == 42
    assert wrapper.get_label(torch.tensor(2)) == 42
    assert wrapper._labels_cache is None


@pytest.mark.utils
@pytest.mark.parametrize("image_key", ["image", "img", "pixel_values"])
def test_hftotv_get_label_works_for_each_supported_image_key(image_key):
    ds = _make_hf_image_dataset(image_key=image_key)
    wrapper = HFtoTV(ds)
    assert wrapper.image_key == image_key
    assert wrapper.get_label(0) == 0


@pytest.mark.utils
def test_hftotv_init_raises_on_missing_image_key():
    ds = datasets.Dataset.from_dict({"foo": [[1.0]], "label": [0]})
    with pytest.raises(ValueError, match="Could not find image key"):
        HFtoTV(ds)
