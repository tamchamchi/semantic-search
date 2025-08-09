class Registry:
    mapping = {
        "semantic_extractor": {},
        "indexer": {},
        "paths": {}
    }

    @classmethod
    def register_semantic_extractor(cls, name):
        def wrap(name_cls):
            from src.semantic_extractor import SemanticExtractor

            assert issubclass(name_cls, SemanticExtractor), (
                "All extractors must inherit 'SemanticExtractor' class."
            )

            if name in cls.mapping["semantic_extractor"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["semantic_extractor"][name]
                    )
                )
            cls.mapping["semantic_extractor"][name] = name_cls
            return name_cls

        return wrap

    @classmethod
    def register_indexer(cls, name):
        def wrap(name_cls):
            from src.indexer import Indexer

            assert issubclass(name_cls, Indexer), (
                "All indexer must inherit 'Indexer' class."
            )

            if name in cls.mapping["indexer"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["indexer"][name]
                    )
                )
            cls.mapping["indexer"][name] = name_cls
            return name_cls
        return wrap

    @classmethod
    def register_path(cls, name, path):
        assert isinstance(path, str), "All path must be str."
        if name in cls.mapping["paths"]:
            raise KeyError("Name '{}' already registered.".format(name))
        cls.mapping["paths"][name] = path

    @classmethod
    def get_semantic_extractor_cls(cls, name):
        return cls.mapping["semantic_extractor"].get(name, None)

    @classmethod
    def list_semantic_extractor(cls):
        return sorted(cls.mapping["semantic_extractor"].keys())

    @classmethod
    def get_indexer_cls(cls, name):
        return cls.mapping["indexer"].get(name, None)

    @classmethod
    def list_indexer(cls):
        return sorted(cls.mapping["indexer"].keys())

    @classmethod
    def get_path(cls, name):
        return cls.mapping["paths"].get(name, None)

    @classmethod
    def list_path(cls):
        return sorted(cls.mapping["paths"].keys())


registry = Registry()
