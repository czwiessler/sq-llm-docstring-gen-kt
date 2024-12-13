import argparse
import os
import sys
import tempfile

from cli import ensure_node_running, ensure_node_has_api, CLIArgsException
from cli.mmt.engine import EngineNode, Engine
from cli.mmt.fileformats import XLIFFFileFormat
from cli.mmt.translation import ModernMTTranslate, EchoTranslate, ModernMTEnterpriseTranslate


class Translator(object):
    def __init__(self, engine):
        self._engine = engine

    def run(self, in_stream, out_stream, threads=None, suppress_errors=False):
        raise NotImplementedError


class XLIFFTranslator(Translator):
    def __init__(self, engine):
        Translator.__init__(self, engine)

    def run(self, in_stream, out_stream, threads=None, suppress_errors=False):
        temp_file = None

        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False) as temp_stream:
                temp_file = temp_stream.name
                temp_stream.write(in_stream.read())

            xliff = XLIFFFileFormat(temp_file, self._engine.target_lang)

            def generator():
                with xliff.reader() as reader:
                    for src_line, _ in reader:
                        yield src_line

            with xliff.writer() as writer:
                self._engine.translate_batch(generator(), lambda r: writer.write(None, r),
                                             threads=threads, suppress_errors=suppress_errors)

            with open(temp_file, 'r', encoding='utf-8') as result:
                out_stream.write(result.read())
        finally:
            if temp_file is not None and os.path.exists(temp_file):
                os.remove(temp_file)


class BatchTranslator(Translator):
    def __init__(self, engine):
        Translator.__init__(self, engine)

    def run(self, in_stream, out_stream, threads=None, suppress_errors=False):
        self._engine.translate_stream(in_stream, out_stream, threads=threads, suppress_errors=suppress_errors)


class InteractiveTranslator(Translator):
    def __init__(self, engine):
        Translator.__init__(self, engine)

        print('\nModernMT Translate command line')

        if isinstance(engine, ModernMTTranslate) and engine.context_vector:
            print('>> Context:', ', '.join(
                ['%s %.1f%%' % (self._memory_to_string(score['memory']), score['score'] * 100)
                 for score in engine.context_vector]))
        else:
            print('>> No context provided.')

        print(flush=True)

    @staticmethod
    def _memory_to_string(memory):
        if isinstance(memory, int):
            return '[' + str(memory) + ']'
        else:
            return memory['name']

    def run(self, in_stream, out_stream, threads=None, suppress_errors=False):
        try:
            while 1:
                out_stream.write('> ')
                line = in_stream.readline()
                if not line:
                    break

                line = line.strip()
                if len(line) == 0:
                    continue

                translation = self._engine.translate_text(line)
                out_stream.write(translation)
                out_stream.write('\n')
                out_stream.flush()
        except KeyboardInterrupt:
            pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Translate text with ModernMT', prog='mmt translate')
    parser.add_argument('text', metavar='TEXT', help='text to be translated (optional)', default=None, nargs='?')
    parser.add_argument('-s', '--source', dest='source_lang', metavar='SOURCE_LANGUAGE', default=None,
                        help='the source language (ISO 639-1). Can be omitted if engine is monolingual.')
    parser.add_argument('-t', '--target', dest='target_lang', metavar='TARGET_LANGUAGE', default=None,
                        help='the target language (ISO 639-1). Can be omitted if engine is monolingual.')

    # Context arguments
    parser.add_argument('--context', metavar='CONTEXT', dest='context',
                        help='A string to be used as translation context')
    parser.add_argument('--context-file', metavar='CONTEXT_FILE', dest='context_file',
                        help='A local file to be used as translation context')
    parser.add_argument('--context-vector', metavar='CONTEXT_VECTOR', dest='context_vector',
                        help='The context vector with format: <document 1>:<score 1>[,<document N>:<score N>]')

    # Mixed arguments
    parser.add_argument('-e', '--engine', dest='engine', help='the engine name, \'default\' will be used if absent',
                        default='default')
    parser.add_argument('--batch', action='store_true', dest='batch', default=False,
                        help='if set, the script will read the whole stdin before send translations to MMT.'
                             'This can be used to execute translation in parallel for a faster translation. ')
    parser.add_argument('--threads', dest='threads', default=None, type=int,
                        help='number of concurrent translation requests.')
    parser.add_argument('--xliff', dest='is_xliff', action='store_true', default=False,
                        help='if set, the input is a XLIFF file.')
    parser.add_argument('--split-lines', dest='split_lines', action='store_true', default=False,
                        help='if set, ModernMT will split input text by carriage-return char')
    parser.add_argument('--quiet', dest='quiet', action='store_true', default=False,
                        help='if set, translation errors are suppressed and an empty translation is returned instead')
    parser.add_argument('--echo', dest='echo', action='store_true', default=False,
                        help='if set, outputs a fake translation coming from an echo server. '
                             'This is useful if you want to test input format validity before '
                             'running the actual translation.')
    parser.add_argument('--api-key', dest='api_key', default=None, help='Use ModernMT Enterprise service instead of '
                                                                        'local engine using the provided API Key')

    args = parser.parse_args(argv)

    engine = Engine(args.engine)

    if args.source_lang is None or args.target_lang is None:
        if len(engine.languages) > 1:
            raise CLIArgsException(parser,
                                   'Missing language. Options "-s" and "-t" are mandatory for multilingual engines.')
        args.source_lang, args.target_lang = engine.languages[0]

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.echo:
        engine = EchoTranslate(args.source_lang, args.target_lang)
    elif args.api_key is not None:
        engine = ModernMTEnterpriseTranslate(args.source_lang, args.target_lang, args.api_key,
                                             context_vector=args.context_vector)
    else:  # local ModernMT engine
        node = EngineNode(Engine(args.engine))
        ensure_node_running(node)
        ensure_node_has_api(node)

        engine = ModernMTTranslate(node, args.source_lang, args.target_lang, context_string=args.context,
                                   context_file=args.context_file, context_vector=args.context_vector,
                                   split_lines=args.split_lines)

    if args.text is not None:
        print(engine.translate_text(args.text.strip()))
    else:
        if args.is_xliff:
            translator = XLIFFTranslator(engine)
        elif args.batch:
            translator = BatchTranslator(engine)
        else:
            translator = InteractiveTranslator(engine)

        try:
            translator.run(sys.stdin, sys.stdout, threads=args.threads, suppress_errors=args.quiet)
        except KeyboardInterrupt:
            pass  # exit
