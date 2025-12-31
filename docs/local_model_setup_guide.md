# Guide: Setting Up Local Qwen3-VL Model Code

This guide explains how to extract and use Qwen3-VL model code locally instead of importing from the `transformers` library. This is essential for your optimization project as it gives you full control over the model architecture.

## Overview

The Qwen3-VL model code is currently in the Hugging Face `transformers` library. To work with it locally, you need to:

1. Extract the model code from transformers
2. Create a local model package structure
3. Modify imports throughout your codebase
4. Handle dependencies and configuration files

## Step-by-Step Approach

### Option 1: Extract from Installed Transformers (Recommended)

#### Step 1: Locate the Model Code in Your Environment

```bash
# Find where transformers is installed
python -c "import transformers; print(transformers.__file__)"

# Navigate to the model code
# Path will be something like:
# .../site-packages/transformers/models/qwen3_vl/
```

#### Step 2: Copy Model Code to Local Repository

Create a local model package structure:

```
Qwen3-VL/
├── qwen3vl_local/          # New local model package
│   ├── __init__.py
│   ├── modeling_qwen3_vl.py
│   ├── modeling_qwen3_vl_moe.py
│   ├── configuration_qwen3_vl.py
│   ├── processing_qwen3_vl.py
│   └── tokenization_qwen3_vl.py
```

**Commands to extract:**

```bash
# Create the local model directory
mkdir -p qwen3vl_local

# Copy from transformers installation
# Adjust path based on your Python environment
cp -r <path_to_site_packages>/transformers/models/qwen3_vl/* qwen3vl_local/
```

#### Step 3: Extract Required Dependencies

The model code depends on other transformers modules. You'll need to copy or adapt:

- `transformers.modeling_utils` → Create minimal wrapper or copy
- `transformers.configuration_utils` → Copy or create wrapper
- `transformers.utils` → Copy utility functions used
- `transformers.modeling_flash_attention_utils` → Copy if using flash attention

**Recommended approach:** Create a `qwen3vl_local/utils/` directory with minimal wrappers that import from transformers for non-critical utilities.

### Option 2: Clone Transformers Repository and Extract

#### Step 1: Clone Hugging Face Transformers

```bash
git clone https://github.com/huggingface/transformers.git transformers_repo
cd transformers_repo
git checkout <version_with_qwen3_vl>  # e.g., v4.57.0 or later
```

#### Step 2: Extract Qwen3-VL Model Code

```bash
# Copy the model directory
cp -r src/transformers/models/qwen3_vl ../Qwen3-VL/qwen3vl_local/

# Copy MoE variant if needed
cp -r src/transformers/models/qwen3_vl_moe ../Qwen3-VL/qwen3vl_local_moe/
```

#### Step 3: Extract Supporting Code

Copy necessary utility modules:

```bash
# Create utils directory
mkdir -p qwen3vl_local/utils

# Copy required utilities (adjust as needed)
cp src/transformers/modeling_utils.py qwen3vl_local/utils/
cp src/transformers/configuration_utils.py qwen3vl_local/utils/
cp src/transformers/utils/generic.py qwen3vl_local/utils/
# Add other dependencies as you discover them
```

### Step 4: Create Local Package Structure

Create `qwen3vl_local/__init__.py`:

```python
"""
Local Qwen3-VL model implementation for optimization research.
"""

from .modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLVisionModel,
    Qwen3VLPreTrainedModel,
)

from .modeling_qwen3_vl_moe import (
    Qwen3VLMoeForConditionalGeneration,
    Qwen3VLMoeModel,
    Qwen3VLMoeVisionModel,
    Qwen3VLMoePreTrainedModel,
)

from .configuration_qwen3_vl import Qwen3VLConfig
from .processing_qwen3_vl import Qwen3VLProcessor

__all__ = [
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLModel",
    "Qwen3VLVisionModel",
    "Qwen3VLPreTrainedModel",
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3VLMoeModel",
    "Qwen3VLMoeVisionModel",
    "Qwen3VLMoePreTrainedModel",
    "Qwen3VLConfig",
    "Qwen3VLProcessor",
]
```

### Step 5: Modify Imports in Your Code

#### Update `src/Qwen3VLTesting.py`:

```python
# OLD:
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# NEW:
import sys
from pathlib import Path

# Add local model to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qwen3vl_local import Qwen3VLForConditionalGeneration
from qwen3vl_local import Qwen3VLProcessor as AutoProcessor
```

#### Update `qwen-vl-finetune/qwenvl/train/train_qwen.py`:

```python
# OLD:
from transformers import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)

# NEW:
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from qwen3vl_local import (
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
```

#### Update `qwen-vl-finetune/qwenvl/train/trainer.py`:

```python
# OLD:
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)

# NEW:
from qwen3vl_local.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLModel,
    apply_rotary_pos_emb,
)
```

### Step 6: Handle Dependencies

Create `qwen3vl_local/utils/__init__.py` with minimal wrappers:

```python
"""
Minimal wrappers for transformers utilities.
For optimization, we import only what's necessary.
"""

# Import critical utilities from transformers
try:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.configuration_utils import PretrainedConfig
    from transformers.utils import logging
except ImportError:
    # Fallback or local implementation
    pass

__all__ = ["PreTrainedModel", "PretrainedConfig", "logging"]
```

### Step 7: Fix Import Paths in Model Code

After copying, you'll need to update imports within the model files:

**In `modeling_qwen3_vl.py`:**

```python
# OLD:
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# NEW (Option A - Use transformers for base classes):
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

# NEW (Option B - Use local if you copied them):
from ..utils.modeling_utils import PreTrainedModel
from ..utils.configuration_utils import PretrainedConfig
```

**Recommendation:** Keep base classes from transformers (Option A) to minimize changes, but make model-specific code local.

### Step 8: Create Setup Script

Create `scripts/setup_local_model.py`:

```python
"""
Script to set up local Qwen3-VL model code.
Run this once to extract model code from transformers.
"""

import os
import shutil
from pathlib import Path
import transformers

def extract_model_code():
    """Extract Qwen3-VL model code from transformers."""
    
    # Find transformers installation
    transformers_path = Path(transformers.__file__).parent
    models_path = transformers_path / "models"
    
    # Create local model directory
    project_root = Path(__file__).parent.parent
    local_model_path = project_root / "qwen3vl_local"
    local_model_path.mkdir(exist_ok=True)
    
    # Copy Qwen3-VL model code
    qwen3_vl_source = models_path / "qwen3_vl"
    if qwen3_vl_source.exists():
        print(f"Copying from {qwen3_vl_source}")
        shutil.copytree(qwen3_vl_source, local_model_path, dirs_exist_ok=True)
        print(f"Copied to {local_model_path}")
    else:
        print(f"Error: {qwen3_vl_source} not found")
        return False
    
    # Copy MoE variant if exists
    qwen3_vl_moe_source = models_path / "qwen3_vl_moe"
    if qwen3_vl_moe_source.exists():
        moe_dest = project_root / "qwen3vl_local_moe"
        shutil.copytree(qwen3_vl_moe_source, moe_dest, dirs_exist_ok=True)
        print(f"Copied MoE variant to {moe_dest}")
    
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Review and update imports in qwen3vl_local/ files")
    print("2. Update your code to import from qwen3vl_local")
    print("3. Test with a simple inference script")
    
    return True

if __name__ == "__main__":
    extract_model_code()
```

## Testing Your Local Setup

Create `tests/test_local_model.py`:

```python
"""Test that local model code works correctly."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qwen3vl_local import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
import torch

def test_model_loading():
    """Test loading model from local code."""
    model_id = "Qwen/Qwen3-VL-2B-Instruct"
    
    print("Loading model from local code...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    print("✓ Model loaded successfully")
    
    processor = Qwen3VLProcessor.from_pretrained(model_id)
    print("✓ Processor loaded successfully")
    
    return model, processor

if __name__ == "__main__":
    test_model_loading()
```

## Important Considerations

### 1. **License Compliance**
- Ensure you comply with the transformers library license (Apache 2.0)
- Keep attribution and license notices

### 2. **Version Compatibility**
- The model code may depend on specific transformers versions
- Document which transformers version you extracted from
- Consider pinning transformers version in requirements

### 3. **Dependency Management**
- You'll still need transformers for:
  - Base classes (PreTrainedModel, PretrainedConfig)
  - Tokenizers (unless you copy those too)
  - Utilities (unless you copy them)
  
- Create `requirements_local.txt`:
```
transformers>=4.57.0  # For base classes and utilities
torch
# ... other dependencies
```

### 4. **Git Strategy**
- Add `qwen3vl_local/` to `.gitignore` initially
- Or commit it if you want version control
- Consider using git submodules if you want to track transformers changes

### 5. **Optimization Modifications**
Once you have local code, you can:
- Modify attention mechanisms directly
- Add quantization hooks
- Implement pruning in model code
- Customize architecture layers
- Add profiling/debugging code

## Recommended Workflow

1. **Week 1:** Extract model code using the setup script
2. **Week 1:** Test that local model works identically to transformers version
3. **Week 2:** Create a comparison test to ensure parity
4. **Week 3+:** Start making optimization modifications

## Troubleshooting

### Import Errors
- Check Python path includes project root
- Verify all dependencies are installed
- Review import statements in copied files

### Missing Dependencies
- Add missing utilities to `qwen3vl_local/utils/`
- Or import from transformers as fallback

### Model Loading Issues
- Ensure config files are accessible
- Check that model weights can still be loaded from HuggingFace
- Verify processor/tokenizer compatibility

## Next Steps

After setting up local model code:

1. Create a baseline test to verify parity
2. Document any modifications you make
3. Set up version control for your model changes
4. Begin implementing optimization techniques

