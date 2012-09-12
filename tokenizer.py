#!/usr/bin/env python3
# -*- coding: utf-8 -*- vim:fenc=utf-8:ft=python:et:sw=4:ts=4:sts=4
"""Tokenizer module.

Basic tokenizer with option to switch token match pattern when certain token is
met.

Module provides following classes:
- Token: Base class for tokens.
- TokenTable: Token storage.
- Tokenizer: Class which does the tokenization and gives access to the stream of
  tokens.
"""

import re
import os
from collections import OrderedDict as Odict


class Token(object):
    """Basic token base class."""
    name = None
    pattern_str = r""

    @classmethod
    def init(cls, name, pattern_str):
        cls.name = name
        cls.pattern_str = pattern_str

    @classmethod
    def info(cls):
        return "class __name__: {}, class.name: {}, pattern: {}".format(cls.__name__, cls.name, repr(cls.pattern_str))

    def __init__(self, pos, value):
        self.id = self.name + "-" + str(pos)
        self.pos = pos
        self.value = value


class TokenTable(object):
    """Token table class for a tokenizer.

    Stores Token classes or subclasses (not instances). Also stores table change
    rules for the tokenizer.

    Attributes:
        name: str. Name for the token table.
        table_change_rules: dict. Rules for the tokenizer, to instruct, after
        which token token table is changed. So a rule is a mapping from a token
        name to TokenTable.
        token_re: re. Compiled regular expression for the matching of the
        tokens. Used by the tokenizer.
        default_token_class: class(Token). For the add_new_token method the
        default token base class.
    """
    def __init__(self, name="token_table"):
        """Intialises TokenTable instance."""
        self.__name = name
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
    ###
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
    def name(self):
        """Name property setter and getter."""
        return self.__name
    @name.setter
    def name(self, new_name):
        self.__name = new_name

    @property
    def table_change_rules(self):
        """table_change_rule property setter/getter/deleter."""
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
    # ###
    # Single rule manipulation
    def get_table_change_rule(self, token_name):
        return self.__table_change_rules[token_name]
    def add_table_change_rule(self, token_name, token_table):
        self.__table_change_rules[token_name] = token_table
    def del_table_change_rule(self, token_name):
        del self.__table_change_rules[token_name]
    # ###
    # Token add/get/remove
    def add_token(self, token):
        assert(issubclass(token, Token))
        self.__table[token.name] = token
        # Invaliadate compiled regex
        self.__token_re = None
    def add_tokens(self, token_iter):
        for t in token_iter:
            self.__table[t.name] = t
        if len(t):
            # Invaliadate compiled regex
            self.__token_re = None
    def add_new_token(self, *vargs, token_subclass=None, **kwargs):
        """
        """
        set_new_class_name = False
        if not token_subclass:
            class c(self.default_token_class):
                pass
            token_subclass = c
            set_new_class_name = True
        assert(issubclass(token_subclass, Token))

        token_subclass.init(*vargs, **kwargs)
        if token_subclass.name in self.__table:
            raise KeyError("Class named '{}' was already in the token table.".format(token_subclass.name))
        if set_new_class_name:
            token_subclass.__name__ = self.default_token_class.__name__ + "-" + token_subclass.name
        self.add_token(token_subclass)
    def remove_token(self, token):
        if isinstance(token, str):
            name = token
        else:
            name == token.name
        self.__table.pop(name)
        # Invaliadate compiled regex
        self.__token_re = None
    def remove_tokens(self):
        self.__table.clear()
        # Invaliadate compiled regex
        self.__token_re = None
    def get_token(self, token_key):
        return self.__table[token_key]
    # ###
    def finalize(self):
        """Should be called after all tokens are added to the table.

            Calls generate_match_re().
        """
        if self.__token_re == None and self.__table:
            self.regenerate_match_re()
    def regenerate_match_re(self):
        """Generates regex to which is used to match tokens in the token table.
        """
        token_re_str = r""
        for token in self.__table.values():
            if token.pattern_str: # Skip tokens with empty pattern
                token_re_str += r"(?P<{}>{})|".format(token.name, token.pattern_str)
        # Remove trailing '|'
        token_re_str = token_re_str[0:-1]
        # Finally compile the regex
        self.__token_re = re.compile(token_re_str, re.MULTILINE)


class TokenizerException(Exception):
    # TODO: add pos and other info to the exception object
    pass


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
    def get_tokens_gen(self, text, yield_eop=True):
        """
        Returns generator for parsed tokens which are instances of Token classes
        from the symbol table. Some tokens can cause symbol table switch using
        'table_change_rules'.

        Parameters:

        - `text`: str. Target text where tokens are parsed.
        - `yield_eop`: bool. Last token returned in case text was fully
          tokenized is end of program token named "EOP".
        """
        def generate_error_msg(pos, text):
            line_start_pos = 0
            if pos != 0:
                line_start_pos = text.rfind('\n', 0, pos)+1
            line_end_pos = text.find(os.linesep, pos, len(text))
            if line_end_pos == -1:
                line_end_pos = len(text)-1
            line_pos = pos - line_start_pos
            emsg = "Tokenizer stopped at pos {} of {} with char {}. The input line was:\n<{}>\n".format(
                    pos, len(text), repr(text[pos]), text[line_start_pos:line_end_pos])
            emsg += " " + line_pos*" " + "^"
            return emsg

        # Initial table
        current_table = self.token_table

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
            yield current_table.get_token(m.lastgroup)(value=m.group(m.lastgroup), pos=pos)

            # After yielding we may change table
            if hasattr(current_table, "table_change_rules"):
                if m.lastgroup in current_table.table_change_rules:
                    current_table = current_table.table_change_rules[m.lastgroup]

        if yield_eop:
            yield current_table.get_token("EOP")(value=None, pos=pos)
