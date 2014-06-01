# -*- coding: utf-8 -*- vim:fenc=utf-8:ft=python:et:sw=4:ts=4:sts=4
"""Basic recursive parser.
"""
import traceback
import sys
import logging
import collections

from . import tokenizer


class Parser(object):
    """Basic recursive parser.

    peek and get_current_token need support from the token generator object.
    BidirGenCache can be used for this.
    """
    def __init__(self, token_generator, original_input=None, log=None):
        """Parser initialization.

        Args:
            token_generator: yieldable. Generator for the tokens which are
                parsed.
            original_input: str. Indexed by position for debug output. Tokens
                position should match this their position in this object.
            log: logging.Logger: Logger to be used for logging.
        """
        self.token_generator = token_generator
        # Every generated token now has quick access to this parser, if token
        # generator object supports generated values attribute modifying through
        # attr_mod (see: BidirGenCache).
        # We could use global or some other token accessable scope also, instead
        # of this "hack" :)
        if hasattr(self.token_generator, "attr_mod"):
            self.token_generator.attr_mod.update([("parser", self)])
        self.childs = []
        self.original_input = original_input
        self.log = log if log != None else logging.getLogger(__name__)
    def __repr__(self):
        def get_str(n, d):
            get_str.string += d*" " + repr(n) + "\n"
            return True,n,None
        get_str.string = ""
        child_visitor(self, get_str)
        return get_str.string
    def __str__(self):
        def get_str(n, d):
            get_str.string += d*" " + str(n) + "\n"
            return True,n,None
        get_str.string = ""
        child_visitor(self, get_str)
        return get_str.string
    def get_token_gen(self):
        return self.token_generator
    def get_current_token(self):
        return self.token_generator.prev()
    def get_next_token(self):
        return next(self.token_generator)
    def next_token(self):
        return next(self.token_generator)
    def peek(self, count=1):
        return self.token_generator.peek(count)
    def expression(self, rbp=0):
        self.log.debug("-------------- rbp={} ---------------".format(rbp))
        self.log.debug("{}: Entering: with current token: {}"
                .format(self.__class__.__name__, self.get_current_token()))
        ct = self.get_current_token()
        # Move left head next non null producing token
        try:
            left = ct.left(self)
        except StopIteration as e:
            self.log.error("{}: Tokens '{}' left() tried to parse beyond last token!"
                    .format(self.__class__.__name__, repr(self.get_current_token())))
            self.log.debug("Traceback:\n" + "".join(traceback.format_stack()))
            sys.exit(1)
        self.log.debug("{}: got left: {}".format(self.__class__.__name__, left))
        self.log.debug("- Right token:{} : {}".format(self.get_current_token().lbp, self.get_current_token()))
        while rbp < self.get_current_token().lbp:
            self.log.debug("--> Current left: {}".format(left))
            self.log.debug("--> Right token:{} : {}".format(self.get_current_token().lbp, self.get_current_token()))
            # TODO: rename return var, not descriptive
            try:
                left = self.get_current_token().right(left, self)
            except SyntaxError as e:
                self.log.error(e)
                self.log.debug("Traceback:\n" + "".join(traceback.format_stack()))
                sys.exit(1)
            except StopIteration as e:
                self.log.error("{}: Tokens '{}' right() tried to parse beyond last token!"
                        .format(self.__class__.__name__, repr(self.get_current_token())))
                self.log.debug("Traceback:\n" + "".join(traceback.format_stack()))
                sys.exit(1)

        self.log.debug("------------------ returning: {}".format(left))
        return left
    def parse(self):
        """Starts parsing the Symbol stream.

        Expects a token/symbol with name "EOP" to used to indicate end-of-program.
        """
        # Read first token as the current token
        if self.token_generator.is_next_first():
            next(self.token_generator)
        while self.get_current_token().name != "EOP":
            self.log.debug("========= Parser level: %s", self.get_current_token())
            new_expression = self.expression()
            self.log.debug("%s: got expression: %s", self.__class__.__name__, new_expression)
            if new_expression:
                self.childs.append(new_expression)
                from_pos = len(self.childs)-5 if len(self.childs) >= 5 else 0
                for i,c in enumerate(self.childs[from_pos:]):
                    self.log.debug("%d: %s", i+from_pos, c)
            else:
                self.log.debug("None or otherwise false new_expression returned.")
        for i,c in enumerate(self.childs):
            self.log.debug("%d: %s", i, c)
    #
    # Utility functions
    #
    def get_dot_graph(self, name="parsed"):
        """Returns dot format graph of the current parse tree as a string.
        """
        def escape(string, echars, dechars=[]):
            """Escapes and double escapes chars with \ in the given "string".

            This is taken from pyworlib.

            Parameters:

            - `string`: str. String which chars are escaped with \.
            - `echars`: [str]. List of strings of length 1 which are escaped from given
              "string".
            - `dechars`: [char]. List of strings of length 1 which are double escaped
              from given "string". Meaning that they are escaped second time only if
              they were orginally escaped in the "string", for example, if dechars is
              ['n'] and the "string" contains r'...\n...' then the string becomes
              '...\\n...'.
            """
            s = list(string)
            for i, c in enumerate(string):
                if dechars and c == '\\' and len(string)-1 >= i+1 and string[i+1] in dechars:
                    s[i] = "\\\\"
                elif c in echars:
                    if c == "\000":
                        s[i] = "\\000"
                    else:
                        s[i] = "\\" + c
            return "".join(s)

        # Make two passes over the graph and generate dot graph to a list of
        # strings.
        def node_vis(node, depth, dot_str_list):
            dot_str_list.append('  {} [label="{}\\n{}"]\n'.format(node.id, node.id, escape(repr(node.value)[1:-1], ['"'], ['n'])))
            return True,node,None
        def edge_vis(node, depth, dot_str_list):
            for child in node.childs:
                dot_str_list.append("    {} -> {}\n".format(node.id, child.id))
            return True,node,None
        dot_str_list = ["Digraph {} {{\n".format(name)]
        child_visitor(self, node_vis, dot_str_list)
        child_visitor(self, edge_vis, dot_str_list)
        dot_str_list.append("}\n")

        return "".join(dot_str_list)

    def process_exp_until(self, parent_token=None, end=[], plain=[], skip=[], asserts=None, nasserts=None):
        """Processes tokens/symbols until one of the end tokens/symbols is met.

        Processed tokens/symbols are added as current tokens childs. Processing
        means using expression() which handels calling left() and right().

        Args:
            end:      [Token()]. List of Tokens on which processing is ended.
            plain:    [Token()]. List of Tokens which are just added as current
                                 tokens childs, without processing them (calling
                                 left()).
            skip:     [Token()]. List of Tokens which are just ignored/skipped.
            asserts:  ( [Token()], [Token()], [Token()] ). 3-tuple of Token
                        lists. Tokens which asserted at start, middle and end.
            nasserts: ( [Token()], [Token()], [Token()] ). 3-tuple of Token
                        lists. Tokens which nasserted at start, middle and end.
        """
        if parent_token == None:
            parent_token = self.get_current_token()
        log_origin = parent_token.__class__.__name__

        # Read next token
        self.next_token()
        # Start asserts
        if nasserts and nasserts[0]:
            parent_token.nassert_token(self.get_current_token(), nasserts[0])
        if asserts and asserts[0]:
            parent_token.assert_token(self.get_current_token(), asserts[0])
        self.log.debug("%s: start asserts OK!" % (log_origin))
        while self.get_current_token().name not in end:
            self.log.debug("%s: current_token_name: %s" % (log_origin, self.get_current_token().name))
            # Middle asserts
            if nasserts and nasserts[1]:
                parent_token.nassert_token(self.get_current_token(), nasserts[1])
            if asserts and asserts[1]:
                parent_token.assert_token(self.get_current_token(), asserts[1])

            # Let's (before) skip "skip" tokens
            while self.get_current_token().name in skip:
                self.next_token()
                continue

            if self.get_current_token().name in plain:
                # Use plain token
                parent_token.childs.append(self.get_current_token())
                self.next_token()
            else:
                # Parse next expression
                exp = self.expression()
                if exp != None: # Skip None expressions
                    parent_token.childs.append(exp)
                self.log.debug("%s: got expression: %s", log_origin, exp)
                # Expression moves to the next token

            # Let's (after) skip "skip" tokens
            while self.get_current_token().name in skip:
                self.next_token()
                continue

        # If just one end token then just skip it, we know what it is.
        if isinstance(end, collections.Iterable) and not isinstance(end, str) and len(end) > 1:
            parent_token.childs.append(self.get_current_token()) # Append token in the end.
        else:
            self.log.debug("%s: skipped end token: %s", log_origin, self.get_current_token().name)

        self.next_token()

        # End asserts
        if nasserts and nasserts[2]:
            parent_token.nassert_token(self.get_current_token(), nasserts[2])
        if asserts and asserts[2]:
            parent_token.assert_token(self.get_current_token(), asserts[2])


class ClassNameAdapter(logging.LoggerAdapter):
    """Prepends class name to the logger messages.
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['classname'], msg), kwargs


class Symbol(tokenizer.Token):
    """Symbol used for recursive parsing.

    Implements right() and left() functions and "lbp" handling.
    """
    lbp = 0
    log = logging.getLogger(__name__)

    @classmethod
    def init(cls, name, pattern_str, lbp=0):
        super().init(name, pattern_str)
        cls.lbp = lbp

    @classmethod
    def info(cls):
        return super().info() + ", lbp: {}".format(cls.lbp)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # For parse tree
        self.log = ClassNameAdapter(self.log, {'classname': self.__class__.__name__})
        self.childs = [] # first, second, third

    def __repr__(self):
        return "{}<{}><pos={}>{}:{}:[{}]".format(
                self.name,
                self.pattern_str,
                self.pos,
                repr(self.value),
                repr(self.subvalues),
                len(self.childs))

    def __str__(self):
        """Human readable representation of the token instance."""
        return self.name + ":" + repr(self.value) + ((":" + repr(self.subvalues)) if self.subvalues else "")

    def left(self, parser=None):
        """Bind from left.

        Function for symbols which binds other symbols from left.

        Implementation of this should return symbol itself and leave next token not
        eaten by this symbol retreavable.
        """
        self.log.warn("Default left implementation called!")
        parser.next_token()
        return self
        #emsg = "Syntax error '{}'".format(self.__class__.__name__)
        #raise SyntaxError(emsg)

    def right(self, left, parser=None):
        """Bind from right.

        Function for symbols which binds other symbols from right.

        Implementation of this should return symbol itself and leave next token not
        eaten by this symbol retreavable.
        """
        emsg = "Unknown operator '{}'".format(self.__class__.__name__)
        raise SyntaxError(emsg)

    ### General util functions
    def get_asserted_expression(self, parser, token_name):
        """Returns expression parsed from the current token, if its name
        matches given, else raises a SyntaxError.
        """
        exp = parser.expression()
        self.assert_token(exp, token_name)
        return exp
    def get_asserted_next_token(self, token_gen, token_name):
        """Returns next token if its name matches given, else raises a SyntaxError.
        """
        next_token = next(token_gen)
        self.assert_token(next_token, token_name)
        return next_token
    def get_nasserted_next_token(self, token_gen, token_name):
        """Returns next token if its name does not match given else raises a SyntaxError.
        """
        next_token = next(token_gen)
        self.nassert_token(next_token, token_name)
        return next_token
    def assert_token(self, token, token_name):
        emsg = "Syntax error '{}': expected '{}' token next (got: {})".format(self.__class__.__name__, token_name, token.id)
        if isinstance(token_name, list):
            if token.name not in token_name:
                raise SyntaxError(emsg)
        elif token.name != token_name:
            raise SyntaxError(emsg)
    def nassert_token(self, token, token_name):
        emsg = "Syntax error '{}': did not expect '{}' token next at position '{}'".format(self.__class__.__name__, token_name, self.pos)
        if isinstance(token_name, list):
            if token.name in token_name:
                raise SyntaxError(emsg)
        elif token.name == token_name:
            raise SyntaxError(emsg)


def child_visitor(parent, actor, *args):
    """Child visitor

    Visists given parents childs (enumeratable object) by calling given actor.
    For the definition of actor function see treeVisitor.

    Example:

    > # Call clean() for Symbols if it exists
    > def call_clean(n, d):
    >     if hasattr(n, "clean") and not n.cleaned:
    >         n.clean()
    >         return False,n,None
    >     return True,n,None
    > parser.child_visitor(call_clean)
    """
    def treeVisitor(node, actor, *args, depth=0, parent=None, node_i=None):
        """Tree node visitor from root to leafs.

        Args:
            actor: (node, depth: int) -> (descend, new_node, new_actor).
                    Actor function which called for every visited node of
                    the tree. Returns 3 values:
                        descend: bool. True if continue to descent to the
                            nodes childs.
                        new_node: object: New object for which the current
                            node is to be replaced.
                        new_actor: Actor function for the childs. Used if
                            ´descend´ was True.
            *args: Non-keyword arguments for the given to the actor.
            depth: int. Given root node depth. Provide if given root is not
                        at depth 0.
            parent: Symbol. Parent node.

        Example:

        > # Print all nodes with increasing indent as depth increseases.
        > # Continue descend if possible, replace current node with "n"
        > # (same node in this case), and no new actor function to replace
        > # current actor function.
        > def print_node(n, d):
        >     print(d*" ", n)
        >     return True,n,None
        > treeVisitor(root, print_node)
        """
        descend, new_node, new_actor = actor(node, depth, *args)
        if hasattr(parent, "childs") and parent and node_i != None:
            parent.childs[node_i] = new_node
        if descend and hasattr(node, "childs"):
            for i,c in enumerate(node.childs):
                treeVisitor(c, new_actor if new_actor else actor, *args, depth=depth+1, parent=node, node_i=i)

    for i,c in enumerate(parent.childs):
        treeVisitor(c, actor, *args, depth=0, parent=parent, node_i=i)
