# -*- coding: utf-8 -*- vim:fenc=utf-8:ft=python:et:sw=4:ts=4:sts=4
"""Tokenizer module.

Basic regular expression based tokenizer with option to switch token match
pattern when certain token is met.

Module provides following classes:
- Token: Base class for tokens.
- TokenTable: Token storage.
- Tokenizer: Class which does the tokenization and gives access to the stream of
  tokens.

Exception classes:
- TokenizerException
- TokenizerRegexpError

Tokenizer takes a TokenTable which contains Token classes. When an input text is
given for tokenization a stream of token instances from the token table classes
is created.
"""


# Try to use <https://pypi.python.org/pypi/regex> instead of python default "re"
# regex library. Python default regex library "re" has now an artificial
# limitation of 100 named groups:
#   File "/usr/lib/python3.3/sre_compile.py", line 505, in compile
#       "sorry, but this version only supports 100 named groups"
#       AssertionError: sorry, but this version only supports 100 named groups
try:
    import regex as re
except ImportError:
    import re
import os
import sys
import logging
from collections import OrderedDict as Odict


class TokenizerException(Exception):
    """Exception occurs when whole input could not be tokenized."""
    # TODO: add pos and other info in the current error message to the exception
    # object.
    pass

class TokenizerRegexpError(Exception):
    """Exception occurs when Tokenizer regexp compilation fails."""
    pass


class Token(object):
    """Basic token base class.

    Class attributes:
        name:        str. Token classes identifying name.
        basename:    str. Parent (base) class' name (after init())
        pattern_str: str. Regex pattern for the tokens of this class as a raw
            string.

    Attributes:
        pos:               int. Position of the token in the input stream.
        value:             *. Value of the token, can be for example a string or
            an int.
        id:                <name>-<pos>
        subvalues:         Token regexp sub matches.
        extra_token_class: class(Token). When token instance is generated by the
            tokenizer generate also instance of this class. This functionality
            helps parsing some languages, though it just moves complexity from
            the parsing side to the tokenizer.
    """
    name = None
    basename = None
    pattern_str = r""
    extra_token_class = None

    @classmethod
    def init(cls, name, pattern_str, ignore=False):
        """Sets class attributes.

        Args:
            name:        str. Token class' identifying name.
            pattern_str: str. Regex pattern for the tokens of this class as a
                raw string.
            ignore:      bool. Is token ignored/skipped during tokenization.
                Default is False.
        """
        cls.name = name
        cls.basename = cls.__base__.__name__
        cls.pattern_str = pattern_str
        cls.ignore = ignore

    @classmethod
    def info(cls):
        """Info gives human readable information about the token class as a string.

        Returns:
            str. Information about the token class.
        """
        return "class __name__: {}, basename: {}, name: {}, pattern: {}".format(cls.__name__, cls.basename, cls.name, repr(cls.pattern_str))

    def __init__(self, pos, value, subvalues=[]):
        """Token instance initializer."""
        self.id = self.name + "-" + str(pos)
        self.pos = pos
        self.value = value
        self.subvalues = subvalues
    def __repr__(self):
        return "{}<{}><pos={}>{}:{}".format(self.name, self.pattern_str, self.pos, repr(self.value), repr(self.subvalues))
    def __str__(self):
        """Human readable representation of the token instance."""
        return self.name + ":" + repr(self.value) + ((":" + repr(self.subvalues)) if self.subvalues else "")


class TokenTable(object):
    """Token table class for a tokenizer.

    Stores Token classes or subclasses (not instances). Also stores table change
    rules for the tokenizer.

    Attributes:
        name:                str. Name for the token table.
        table_change_rules:  {str: TokenTable()}. Rules for the tokenizer, to
            instruct, after which token token table is changed. So a rule is a
            mapping from a token name to TokenTable.
        token_re:            re. Compiled regular expression for the matching of
            the tokens. Used by the tokenizer.
        default_token_class: class(Token). For the add_new_token method the
            default token base class.
    """
    def __init__(self, name="token_table"):
        """Intialises TokenTable instance."""
        self.name = name
        # Maps "NAME/ID" to a Token (class object not instance) in ordered fashion.
        self.__table = Odict()
        # Table change rules pairs "TOKEN_NAME" with another TokenTable object.
        # For example:
        # > comment_table = TokenTable()
        #   .. add tokens to the comment_table ..
        # > table_change_rule = { 'COMMENT': comment_table }
        # > comment_table.put_table_change_rules(table_change_rule)
        self.__table_change_rules = {}

        # Token matcher regex is generated from the table
        self.__token_re = None

        # Default token class is used with add_new_token method
        self.__default_token_class = Token
    def __str__(self):
        return "<{},{}>{}".format(self.__token_re.pattern if self.__token_re else "None", self.__table_change_rules, self.__table)
    def info(self):
        istr = "Table '{}':\n".format(self.name)
        if self.__table:
            istr += " Tokens:\n"
        else:
            istr += " Empty token table\n"
        for k,v in self.__table.items():
            istr += "  [{:<13}]: {}\n".format(k, v.info())
        if self.__table_change_rules:
            istr += " Table change rules: <from token class to (->) token table>\n"
            for k,v in self.__table_change_rules.items():
                istr += "  {} -> {}\n".format(k, v.name)
        return istr
    ### Properties
    @property
    def table_change_rules(self):
        """table_change_rules property setter/getter/deleter."""
        return self.__table_change_rules
    @table_change_rules.setter
    def table_change_rules(self, rules):
        self.__table_change_rules = rules
    @table_change_rules.deleter
    def table_change_rules(self):
        self.__table_change_rules.clear()

    @property
    def token_re(self):
        """token_re property getter."""
        return self.__token_re

    @property
    def default_token_class(self):
        """default_token_class property setter/getter/deleter."""
        return self.__default_token_class
    @default_token_class.setter
    def default_token_class(self, token_class):
        assert(issubclass(token_class, Token))
        self.__default_token_class = token_class
    @default_token_class.deleter
    def default_token_class(self):
        self.__default_token_class = Token

    # Single rule manipulation
    def get_table_change_rule(self, token_name):
        """Return table change rule with the given token name."""
        return self.__table_change_rules[token_name]
    def add_table_change_rule(self, token_name, token_table):
        """Add table change rule from a token name to a token table."""
        self.__table_change_rules[token_name] = token_table
    def del_table_change_rule(self, token_name):
        """Delete table change rule with the given token name."""
        del self.__table_change_rules[token_name]

    # Token add/get/remove
    def add_token(self, token):
        """Add given token to the token table.

        Args:
            token. class(Token). Token class to be added to the token table.
        """
        assert(issubclass(token, Token))
        self.__table[token.name] = token
        # Invaliadate compiled regex
        self.__token_re = None
    def add_tokens(self, token_iter):
        """Add given tokens from iterable object to the token table.

        Args:
            token_iter. [class(Token)]. Token classes to be added to the token
                table.
        """
        for t in token_iter:
            self.__table[t.name] = t
        if len(token_iter):
            # Invalidate compiled regex
            self.__token_re = None
    def add_new_token(self, *vargs, token_subclass=None, **kwargs):
        """Create and add a new token class to the token table.

        New token classes are child classes of given token_subclass or if not
        given then default_token_class is used. In any case new class is always
        created.

        Args:
            token_subclass: class(Token). Create the new token class as child
                class of this. If none given token tables default_token_class
                attribute is used.
            *vargs:         *. Passed to the token_subclass classmethod init()
                which is inherited from the Token base class.
            **kwargs:       *. Passed to the token_subclass classmethod init()
                which is inherited from the Token base class.
        """
        if token_subclass == None:
            token_subclass = self.default_token_class

        class new_token_class(token_subclass):
            pass

        assert(issubclass(new_token_class, Token))

        new_token_class.init(*vargs, **kwargs)

        # XXX: what is new_token_class.name
        if new_token_class.name in self.__table:
            raise KeyError("Class named '{}' was already in the token table.".format(new_token_class.name))

        new_token_class.__name__ = token_subclass.__name__ + "-" + new_token_class.name

        self.add_token(new_token_class)
    def remove_token(self, token):
        """Remove a token from the token table.

        Args:
            token. str|class(Token). Token to be removed from the token table.
                Can be given as a name or a class.
        """
        if isinstance(token, str):
            name = token
        else:
            name == token.name
        self.__table.pop(name)
        # Invaliadate compiled regex
        self.__token_re = None
    def remove_tokens(self):
        """Remove all tokens from the token table."""
        self.__table.clear()
        # Invaliadate compiled regex
        self.__token_re = None
    def get_token(self, token_name):
        """Get token with given name.

        Args:
            token_name: str. Name of the token to get.
        """
        return self.__table[token_name]
    def get_tokens(self):
        """Get all tokens.

        Returns:
            [class(Token)]. Tokens in a list.
        """
        return list(self.__table.values())
    # ###
    def finalize(self):
        """Should be called after all tokens are added to the table.

        Calls generate_match_re() if token table is not empty and token_re has
        not been previously compiled.

        Raises TokenizerRegexpError if regexp compilation fails.
        """
        if self.__token_re == None and self.__table:
            self.regenerate_match_re()
    def regenerate_match_re(self):
        """Generates regex to which is used to match tokens in the token table.

        The regex is generated from the tokens stored in the token table. It
        needs to be regenerated manually every time a token is added iff token
        table is used by the tokenizer in between.

        Modifies attributes:
            self.__token_re

        Raises TokenizerRegexpError if regexp compilation fails.
        """
        def find_broken_token_regex():
            """Tries to find which token regex is broken.

            Returns:
                (str, str). Tuple of token name and token regex.
            """
            trs = r""
            for token in self.__table.values():
                if token.pattern_str: # Skip tokens with empty pattern
                    trs += r"(?P<{}>{})".format(token.name, token.pattern_str)
                    try:
                        re.compile(trs, re.MULTILINE)
                    except Exception:
                        return (token.name, token.pattern_str)
                    trs += r"|"

        token_re_str = r""
        for token in self.__table.values():
            if token.pattern_str: # Skip tokens with empty pattern
                token_re_str += r"(?P<{}>{})|".format(token.name, token.pattern_str)
        # Remove trailing '|'
        token_re_str = token_re_str[0:-1]
        # Finally try to compile the regex
        try:
            self.__token_re = re.compile(token_re_str, re.MULTILINE)
        except Exception as e:
            tb = sys.exc_info()[2]
            token_name, broken_regex = find_broken_token_regex()
            emsg = str(e) + " With token '{}' and regexp: '{}' and whole regexp: {}".format(token_name, broken_regex, token_re_str)
            raise TokenizerRegexpError(emsg).with_traceback(tb)


class Tokenizer(object):
    """Tokenizer class for input text tokenization.

    Uses TokenTables to provide a compiled regex to match for the tokens.

    Given token tables can have "table_change_rules" attribute. Attribute should
    act like dict type with token names as keys and token tables as values. For
    example, {'COMMENT': comment_table} change rule would mean that after
    comment token is found, switch to comment_table.

    All such context tables should be given as argument in the context_tables list
    at the moment.
    XXX: fix this, tables can be parsed from the rules.

    Attributes:
        token_table: class(TokenTable): Main token table, which is used as
            tokenization is started.
    """
    def __init__(self, token_table, context_tables=[]):
        """Tokenizer init.

        TODO: get tables from the rules automatically.
        """
        # Set initial token/symbol table
        self.token_table = token_table
        self.token_table.finalize()

        self.context_tables = context_tables
        for ct in self.context_tables:
            ct.finalize()
    def __str__(self):
        return super().__str__() + "<" + str(self.token_table) + ">"
    def info(self):
        # TODO: return more elaborate information string than __str__.
        pass
    def get_python_code(self):
        """Experimental python code generator for the tokenizer.

        TODO:
        Returns code for a module which provides tok_gen(input_text) generator
        function.

        Returns:
            str. Python code as a string.
        """
        log = logging.getLogger(__name__)
        if self.context_tables:
            log.error("Not implemented with context tables.")
            return ""
        return self.token_table.token_re.pattern
    def get_tokens_gen(self, text, yield_eop=True):
        """
        Returns generator for parsed tokens which are instances of Token classes
        from the symbol table. Some tokens can cause symbol table switch using
        'table_change_rules'.

        Note the magic token named "EOP" if yield_eop is True! If yield_eop is
        True and no "EOP" named token exists in main token table, then the "EOP"
        token is added.

        Args:
            text:      str. Target text where tokens are parsed.
            yield_eop: bool. Last token returned in case text was fully
                tokenized is end of program token named "EOP".

        Raises:
            TokenizerException: if the whole input was not tokenized.
        """
        def generate_error_msg(pos, text):
            line_start_pos = 0
            line_num = 1
            if pos != 0:
                line_start_pos = text.rfind(os.linesep, 0, pos)+1
                line_num = text.count(os.linesep, 0, pos)+1
            line_end_pos = text.find(os.linesep, pos, len(text))
            if line_end_pos == -1:
                line_end_pos = len(text)-1
            line_pos = pos - line_start_pos
            emsg = "Tokenizer stopped at pos {}/{} in line {} with char {}, with table {}. The input line was:\n<{}>\n".format(
                    pos, len(text), line_num, repr(text[pos]), current_table.name, text[line_start_pos:line_end_pos])
            emsg += " " + line_pos*" " + "^"
            return emsg
        def get_sub_matches(groups):
            sub_matches = []
            i = 0
            while len(groups) > i:
                # Group matches don't have to exists, for example in:
                # r"[ab](test)?=(.+)" the (test)? group might be None.
                if groups[i] != None:
                    sub_matches.append(groups[i])
                i += 1
            return sub_matches

        log = logging.getLogger(__name__)

        # Check that EOP token is given, add it if not
        if yield_eop:
            try:
                self.token_table.get_token("EOP")
            except KeyError:
                self.token_table.add_new_token("EOP", r"")

        # Initial table
        current_table = self.token_table

        # Adding new token invalidates matchin regex, so regenerate it, if this
        # has happened.
        if not current_table.token_re:
            current_table.regenerate_match_re()

        pos = 0
        while True:
            m = current_table.token_re.match(text, pos)
            if not m:
                if pos != len(text):
                    # If could not tokenize whole input text
                    raise TokenizerException(generate_error_msg(pos, text))
                break
            pos = m.end()

            # Yield instance of a token class from the token_table
            token_class = current_table.get_token(m.lastgroup)

            # If token regexp had sub matches, store them also with the token
            token_subvalues = get_sub_matches(m.groups()[m.lastindex:])
            # TODO: print debug log about sub matches
            # print("tokenizer:\n", m.group(m.lastgroup), token_subvalues, m.lastindex, m.groups())

            # Finally yield if not ignored token
            if not token_class.ignore:
                yield token_class(value=m.group(m.lastgroup), pos=pos, subvalues=token_subvalues)
                if token_class.extra_token_class:
                    # Yield extra token marker
                    yield token_class.extra_token_class(pos=pos, value=None)

            # After yielding or ignoring we may change table
            if hasattr(current_table, "table_change_rules"):
                if m.lastgroup in current_table.table_change_rules:
                    current_table = current_table.table_change_rules[m.lastgroup]
                    # Check that table has been initialized (token regex
                    # compiled)
                    if not current_table.token_re:
                        current_table.regenerate_match_re()
                    log.debug("Current token table change: {} -> {}".format(m.lastgroup, current_table.name))

        if yield_eop:
            yield current_table.get_token("EOP")(value=None, pos=pos)
