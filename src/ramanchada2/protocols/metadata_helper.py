class MetadataExtractor:
    def extract(self, spectrum, filename=None):
        raise NotImplementedError("Subclasses should implement this method.")


class TemplateMetadataExtractor(MetadataExtractor):
    def __init__(self, template):
        self.template = template

    def extract(self, spectrum, filename=None):
        return {key: spectrum.get(key) for key in self.template}


class FilenameMetadataExtractor(MetadataExtractor):
    def extract(self, spectrum, filename):
        return {"filename": filename}


class SpectrumMetadataExtractor(MetadataExtractor):
    def extract(self, spectrum, filename=None):
        return spectrum.get_metadata()


class ChainedMetadataExtractor(MetadataExtractor):
    def __init__(self, *extractors):
        self.extractors = extractors

    def extract(self, spectrum, filename=None):
        metadata = {}
        for extractor in self.extractors:
            metadata.update(extractor.extract(spectrum, filename))
        return metadata
