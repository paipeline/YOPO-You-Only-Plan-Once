"""
Utility functions for YOPO agents.
"""

import re
import os
from dotenv import load_dotenv
import io
from typing import BinaryIO, Any
import traceback
import requests
import tempfile

import camelot
from openai import OpenAI
from markitdown import MarkItDown
from markitdown.converters import PdfConverter
from markitdown.converters import AudioConverter
from markitdown.converters._pdf_converter import _dependency_exc_info
from markitdown.converters._exiftool import exiftool_metadata
from markitdown._stream_info import StreamInfo
from markitdown._base_converter import DocumentConverterResult
from markitdown._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE
import pdfminer
from litellm import transcription

load_dotenv(verbose=True)


def json_format_parser(text: str) -> str:
    """
    Parse JSON content from text that may contain markdown code blocks or other formatting.
    
    Args:
        text: Text that may contain JSON in various formats
        
    Returns:
        str: Clean JSON string
    """
    # Remove markdown code blocks
    if "```json" in text:
        # Extract content between ```json and ```
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    if "```" in text:
        # Extract content between ``` blocks
        match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Try to find JSON-like content
    # Look for content between curly braces
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()
    
    # If no JSON found, return original text
    return text.strip()


def read_tables_from_stream(file_stream):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
        temp_pdf.write(file_stream.read())
        temp_pdf.flush()
        tables = camelot.read_pdf(temp_pdf.name, flavor="lattice")
        return tables

def transcribe_audio(file_stream):
    response = transcription(model=os.getenv("AUDIO_MODEL_NAME"), file=file_stream).json()
    result = response.get("text", "No transcription available.")

    return result

class AudioWhisperConverter(AudioConverter):
    def convert(
            self,
            file_stream: BinaryIO,
            stream_info: StreamInfo,
            **kwargs: Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        md_content = ""

        # Add metadata
        metadata = exiftool_metadata(
            file_stream, exiftool_path=kwargs.get("exiftool_path")
        )
        if metadata:
            for f in [
                "Title",
                "Artist",
                "Author",
                "Band",
                "Album",
                "Genre",
                "Track",
                "DateTimeOriginal",
                "CreateDate",
                # "Duration", -- Wrong values when read from memory
                "NumChannels",
                "SampleRate",
                "AvgBytesPerSec",
                "BitsPerSample",
            ]:
                if f in metadata:
                    md_content += f"{f}: {metadata[f]}\n"

        # Figure out the audio format for transcription
        if stream_info.extension == ".wav" or stream_info.mimetype == "audio/x-wav":
            audio_format = "wav"
        elif stream_info.extension == ".mp3" or stream_info.mimetype == "audio/mpeg":
            audio_format = "mp3"
        elif (
                stream_info.extension in [".mp4", ".m4a"]
                or stream_info.mimetype == "video/mp4"
        ):
            audio_format = "mp4"
        else:
            audio_format = None

        # Transcribe
        if audio_format:
            try:
                transcript = transcribe_audio(file_stream)
                if transcript:
                    md_content += "\n\n### Audio Transcript:\n" + transcript
            except MissingDependencyException:
                pass

        # Return the result
        return DocumentConverterResult(markdown=md_content.strip())

class PdfWithTableConverter(PdfConverter):
    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        # Check the dependencies
        if _dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".pdf",
                    feature="pdf",
                )
            ) from _dependency_exc_info[
                1
            ].with_traceback(  # type: ignore[union-attr]
                _dependency_exc_info[2]
            )

        assert isinstance(file_stream, io.IOBase)  # for mypy

        tables = read_tables_from_stream(file_stream)
        num_tables = tables.n
        # if num_tables == 0:
        return DocumentConverterResult(
            markdown=pdfminer.high_level.extract_text(file_stream),
            )
        # else:
        #     markdown_content = pdfminer.high_level.extract_text(file_stream)
        #     table_content = ""
        #     for i in range(num_tables):
        #         table = tables[i].df
        #         table_content += f"Table {i + 1}:\n" + table.to_markdown(index=False) + "\n\n"
        #     markdown_content += "\n\n" + table_content
        #     return DocumentConverterResult(
        #         markdown=markdown_content,
        #     )

class MarkitdownConverter():
    def __init__(self,
                 use_llm: bool = False,
                 timeout: int = 30):

        self.timeout = timeout
        self.use_llm = use_llm

        if use_llm:
            client = OpenAI()
            self.client = MarkItDown(
                enable_plugins=True,
                llm_client=client,
                llm_model=os.getenv("MARKTIDOWN_MODEL_NAME")
            )
        else:
            self.client = MarkItDown(
                enable_plugins=True,
            )

        removed_converters = [
            PdfConverter, AudioConverter, 
        ]

        self.client._converters = [
            converter for converter in self.client._converters
            if not isinstance(converter.converter, tuple(removed_converters))
        ]

        self.client.register_converter(PdfWithTableConverter())
        self.client.register_converter(AudioWhisperConverter())

    def convert(self, source: str, **kwargs: Any):
        try:
            result = self.client.convert(
                source,
                **kwargs)
            return result
        except Exception as e:
            
            print(f"❌ Error in MarkitdownConverter.convert: {str(e)}")
            print(f"📍 Traceback:")
            traceback.print_exc()

            return None
        

if __name__ == "__main__":
    vert = MarkitdownConverter()
    breakpoint()
    result = vert.convert("/Users/jakcieshi/Desktop/Home/Projects/FreeLan/PARA_Cheat_Sheet.pdf")