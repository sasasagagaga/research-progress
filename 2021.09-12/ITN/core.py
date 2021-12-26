
# For random generators
random_seed = 179

# Paths
# TODO: Rename _paths -> _dir
# TODO: Remove language variants (like in run_logs_dir)
language_for_human = {
    "ru": "russian",
    "en": "english"
}

data_paths = {
    "ru": "data/russian/",
    "en": "data/english/"
}

models_paths = {
    "ru": "models/russian/",
    "en": "models/english/"
}

run_logs_dir = "run_logs"

# TODO: Использовать по возможности __slots__ в классах
# TODO: Переместить def count_parameters(model) сюда?
