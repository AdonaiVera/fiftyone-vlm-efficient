import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types

# Common UI utilities
def _model_choice_inputs(ctx, inputs):
    model_paths = [
        "microsoft/Florence-2-base",
        "microsoft/Florence-2-large",
        "microsoft/Florence-2-base-ft",
        "microsoft/Florence-2-large-ft",
    ]

    radio_group = types.RadioGroup()
    for model_path in model_paths:
        radio_group.add_choice(model_path, label=model_path)

    inputs.enum(
        "model_path",
        radio_group.values(),
        label="Model path",
        description="The model checkpoint to use for the operation",
        required=False,
        view=types.DropdownView(),
    )

    _model_download_check_inputs(ctx, inputs)

def _model_download_check_inputs(ctx, inputs):
    model_choice = ctx.params.get("model_path", None)
    if model_choice is None:
        return

    base_path = "~/.cache/huggingface/hub/models--"
    model_path_formatted = base_path + model_choice.replace("/", "--")
    model_dir = os.path.expanduser(model_path_formatted)

    if not os.path.exists(model_dir):
        description = (
            f"Model {model_choice} has not been downloaded. The model will be "
            "downloaded automatically the first time you run this operation."
            "Please be aware that this may take some time."
        )
        inputs.view(
            "model_download_warning",
            types.Warning(
                label="Model not downloaded", description=description
            ),
        )

def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )

def _handle_calling(
        uri, 
        sample_collection, 
        model_path,
        operation,
        output_field,
        delegate=False,
        **kwargs
        ):
    """Helper function to handle operator calling via SDK."""
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_path=model_path,
        operation=operation,
        output_field=output_field,
        delegate=delegate,
        **kwargs
        )
    return foo.execute_operator(uri, ctx, params=params)

# SIEVE-specific utilities
def _sieve_captioning_model_inputs(ctx, inputs):
    """Add inputs for SIEVE captioning model selection."""
    captioning_models = [
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large",
        "microsoft/git-base",
        "microsoft/git-large"
    ]
    
    model_radio = types.RadioGroup()
    for model in captioning_models:
        model_radio.add_choice(model, label=model)
        
    inputs.enum(
        "captioning_model",
        model_radio.values(),
        label="Captioning Model",
        description="Choose the image captioning model to use",
        default=captioning_models[0]
    )

def _sieve_alt_text_field_inputs(ctx, inputs):
    """Add inputs for SIEVE alt text field selection."""
    if ctx.dataset is not None:
        string_fields = list(
            ctx.dataset.get_field_schema(ftype=fo.StringField).keys()
        )
        if string_fields:
            field_radio = types.RadioGroup()
            for field in string_fields:
                field_radio.add_choice(field, label=field)
                
            inputs.enum(
                "alt_text_field",
                field_radio.values(),
                label="Alt Text Field",
                description="Field containing original text/alt-text"
            )

def _sieve_encoder_model_inputs(ctx, inputs):
    """Add inputs for SIEVE sentence encoder model selection."""
    encoder_models = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    
    encoder_radio = types.RadioGroup()
    for model in encoder_models:
        encoder_radio.add_choice(model, label=model)
        
    inputs.enum(
        "sentence_encoder",
        encoder_radio.values(),
        label="Sentence Encoder",
        description="Model for computing semantic similarity",
        default=encoder_models[0]
    )

def _sieve_medium_phrases_inputs(ctx, inputs):
    """Add inputs for SIEVE medium phrases."""
    inputs.list(
        "medium_phrases",
        types.String(),
        default=["an image of", "photo of", "picture of", "there is"],
        label="Medium Phrases",
        description="Common phrases to be masked"
    )

def _sieve_fusion_inputs(ctx, inputs):
    """Add inputs for SIEVE CLIP fusion."""
    inputs.bool(
        "use_fusion",
        default=False,
        label="Use CLIP Fusion",
        description="Enable fusion with CLIPScore"
    )
    
    # Fusion weight (only shown if use_fusion is True)
    if ctx.params.get("use_fusion", False):
        inputs.float(
            "fusion_weight",
            default=0.5,
            label="Fusion Weight",
            description="Weight for CLIP fusion (0-1)",
            min=0.0,
            max=1.0
        )

def _sieve_k_fraction_inputs(ctx, inputs):
    """Add inputs for SIEVE k-fraction."""
    inputs.float(
        "k_fraction",
        default=0.2,
        label="K Fraction",
        description="Fraction of top-ranked samples to keep (0-1)",
        min=0.0,
        max=1.0
    )

def _sieve_num_captions_inputs(ctx, inputs):
    """Add inputs for SIEVE number of captions."""
    inputs.int(
        "num_captions",
        default=4,
        label="Number of Captions",
        description="Number of captions to generate via nucleus sampling"
    )
