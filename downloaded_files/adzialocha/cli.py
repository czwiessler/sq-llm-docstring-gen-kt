import os
import sys

import click

from tomomibot.utils import health_check


CONTEXT_SETTINGS = dict(auto_envvar_prefix='COMPLEX')

# Disable debugging logs of tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Context(object):

    def __init__(self):
        self.verbose = False

    def log(self, msg, *args):
        """Logs a message to stderr."""
        if args:
            msg %= args
        click.echo(msg)

    def vlog(self, msg, *args):
        """Logs a message to stderr only if verbose is enabled."""
        if self.verbose:
            self.log(msg, *args)

    def elog(self, msg, *args):
        """Logs an error message to stderr and exists."""
        if args:
            msg %= args
        click.echo(click.style('Error: %s' % msg, fg='red'), file=sys.stderr)
        sys.exit(1)


commands_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'commands'))

pass_context = click.make_pass_decorator(Context, ensure=True)


class Console(click.MultiCommand):

    def list_commands(self, ctx):
        commands = []
        for file_name in os.listdir(commands_dir):
            if file_name.endswith('.py') and \
                    not file_name.startswith('__'):
                commands.append(file_name[:-3])
        commands.sort()
        return commands

    def get_command(self, ctx, name):
        try:
            if sys.version_info[0] == 2:
                name = name.encode('ascii', 'replace')
            mod = __import__('tomomibot.commands.' + name,
                             None, None, ['cli'])
        except ImportError:
            return
        return mod.cli


@click.command(cls=Console, context_settings=CONTEXT_SETTINGS)
@click.option('-v', '--verbose', is_flag=True,
              help='Enables verbose mode.')
@pass_context
def cli(ctx, verbose):
    """Artificial intelligence bot for live voice improvisation."""
    ctx.verbose = verbose

    # Make a startup health check to make sure everything is ready
    health_check()
