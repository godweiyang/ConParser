class PhraseTree(object):
    def __init__(
        self,
        symbol=None,
        children=[],
        sentence=[],
        leaf=None,
    ):
        self.symbol = symbol
        self.children = children
        self.sentence = sentence
        self.leaf = leaf
        self._str = None

    def __str__(self):
        if self._str is None:
            if len(self.children) != 0:
                childstr = ' '.join(str(c) for c in self.children)
                self._str = '({} {})'.format(self.symbol, childstr)
            else:
                self._str = '({} {})'.format(
                    self.sentence[self.leaf][1],
                    self.sentence[self.leaf][0],
                )
        return self._str

    def propagate_sentence(self, sentence):
        self.sentence = sentence
        for child in self.children:
            child.propagate_sentence(sentence)

    @staticmethod
    def parse(line):
        line += " "
        sentence = []
        _, t = PhraseTree._parse(line, 0, sentence)

        if t.symbol == 'TOP' and len(t.children) == 1:
            t = t.children[0]

        return t

    @staticmethod
    def _parse(line, index, sentence):
        index += 1
        symbol = None
        children = []
        leaf = None
        while line[index] != ')':
            if line[index] == '(':
                index, t = PhraseTree._parse(line, index, sentence)
                children.append(t)
            else:
                if symbol is None:
                    rpos = min(line.find(' ', index), line.find(')', index))
                    symbol = line[index:rpos]
                    index = rpos
                else:
                    rpos = line.find(')', index)
                    word = line[index:rpos]
                    sentence.append((word, symbol))
                    leaf = len(sentence) - 1
                    index = rpos

            if line[index] == " ":
                index += 1

        t = PhraseTree(
            symbol=symbol,
            children=children,
            sentence=sentence,
            leaf=leaf,
        )

        return (index + 1), t

    def left_span(self):
        """
        return:
            left bound of the span
        """
        try:
            return self._left_span
        except AttributeError:
            if self.leaf is not None:
                self._left_span = self.leaf
            else:
                self._left_span = self.children[0].left_span()
            return self._left_span

    def right_span(self):
        """
        return:
            right bound of the span
        """
        try:
            return self._right_span
        except AttributeError:
            if self.leaf is not None:
                self._right_span = self.leaf
            else:
                self._right_span = self.children[-1].right_span()
            return self._right_span

    @staticmethod
    def load_treefile(fname):
        trees = []
        for line in open(fname):
            try:
                t = PhraseTree.parse(line)
                trees.append(t)
            except:
                print(line)
        return trees

    def enclosing(self, left, right, equal=True):
        """
        return:
            the smallest span that >= span(left, right) if equal=True,
            else the smallest span that > span(left, right).
        """
        for child in self.children:
            l = child.left_span()
            r = child.right_span()
            if (l <= left) and (right <= r):
                if not equal and (l == left) and (r == right):
                    break
                return child.enclosing(left, right, equal)

        return self

    def span_labels(self, left, right):
        """
        return:
            list of symbols of span(left, right) in top-down order if span(left, right) exists,
            else empty list.
        """
        crossing = False
        if self.leaf is not None:
            return [], crossing

        if (self.left_span() == left) and (self.right_span() == right):
            result = [self.symbol]
        else:
            result = []

        for child in self.children:
            l = child.left_span()
            r = child.right_span()
            if (l <= left) and (right <= r):
                child_labels, crossing = child.span_labels(left, right)
                result.extend(child_labels)
                break
            if left < l <= right < r or l < left <= r < right:
                crossing = True

        return result, crossing

    def span_splits(self, left, right):
        """
        return:
            list of splits of the smallest span >= span(left, right) between left and right.
        """
        subtree = self.enclosing(left, right, equal=True)
        return [
            child.left_span()
            for child in subtree.children
            if left < child.left_span() <= right
        ]

    def parent_rights(self, left, split, right_bound):
        subtree = self.enclosing(left, split, equal=False)
        right = subtree.right_span()
        if right == split:
            right = right_bound
        else:
            right = min(right, right_bound)
        if split + 1 == right:
            return [right]
        subtree = self.enclosing(split + 1, right, equal=True)
        return [
            child.right_span()
            for child in subtree.children
            if split < child.right_span() <= right
        ]
