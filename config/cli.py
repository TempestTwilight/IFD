"""Helper to convert argparse args to config overrides."""


def args_to_overrides(args) -> dict:
    """Convert argparse namespace to config override dict.

    Args:
        args: argparse.Namespace from parse_args()

    Returns:
        Dictionary suitable for load_config(overrides=...)
    """
    overrides: dict = {}

    # Seed (always present)
    if hasattr(args, "seed"):
        overrides["seed"] = args.seed

    # Data section
    data = {}
    if hasattr(args, "data_dir") and args.data_dir:
        data["data_dir"] = args.data_dir
    if hasattr(args, "nrows") and args.nrows is not None:
        data["nrows"] = args.nrows
    if hasattr(args, "num_clients") and args.num_clients:
        data["num_clients"] = args.num_clients
    if data:
        overrides["data"] = data

    # Training section
    training = {}
    if hasattr(args, "batch_size") and args.batch_size:
        training["batch_size"] = args.batch_size
    if hasattr(args, "epochs_per_round") and args.epochs_per_round:
        training["epochs_per_round"] = args.epochs_per_round
    if hasattr(args, "lr") and args.lr:
        training["lr"] = args.lr
    if training:
        overrides["training"] = training

    # FL section
    fl = {}
    if hasattr(args, "num_rounds") and args.num_rounds:
        fl["num_rounds"] = args.num_rounds
    if fl:
        overrides["fl"] = fl

    # Attack section
    attack = {}
    if hasattr(args, "num_adversaries") and args.num_adversaries:
        attack["num_adversaries"] = args.num_adversaries
    if hasattr(args, "attack_type") and args.attack_type:
        attack["attack_type"] = args.attack_type
    if attack:
        overrides["attack"] = attack

    # Defense section
    defense = {}
    if hasattr(args, "disable_layer1") and args.disable_layer1:
        defense["disable_layer1"] = True
    if hasattr(args, "disable_layer2") and args.disable_layer2:
        defense["disable_layer2"] = True
    if hasattr(args, "disable_layer3") and args.disable_layer3:
        defense["disable_layer3"] = True
    if defense:
        overrides["defense"] = defense

    # Baseline section
    baseline = {}
    if hasattr(args, "baseline") and args.baseline:
        baseline["baseline"] = args.baseline
    if baseline:
        overrides["baseline"] = baseline

    # Output section
    output = {}
    if hasattr(args, "results_dir") and args.results_dir:
        output["results_dir"] = args.results_dir
    if hasattr(args, "save_model") and args.save_model:
        output["save_model"] = args.save_model
    if hasattr(args, "run_label") and args.run_label:
        output["run_label"] = args.run_label
    if output:
        overrides["output"] = output

    return overrides
