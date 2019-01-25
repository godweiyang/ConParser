from .trees import PhraseTree


class State(object):
    def __init__(self, n):
        self.n = n
        self.i = 0
        self.stack = []

    def can_shift(self):
        return (self.i < self.n)

    def can_combine(self):
        return (len(self.stack) > 1)

    def shift(self):
        j = self.i
        treelet = PhraseTree(leaf=j)
        self.stack.append((j, j, [treelet]))
        self.i += 1

    def combine(self):
        (_, right, treelist0) = self.stack.pop()
        (left, _, treelist1) = self.stack.pop()
        self.stack.append((left, right, treelist1 + treelist0))

    def label(self, nonterminals=[]):
        for nt in nonterminals:
            (left, right, trees) = self.stack.pop()
            tree = PhraseTree(symbol=nt, children=trees)
            self.stack.append((left, right, [tree]))

    def take_action(self, action):
        if action == 'sh':
            self.shift()
        elif action == 'comb':
            self.combine()
        elif action == 'none':
            return
        elif action.startswith('label-'):
            self.label(action[6:].split('-'))
        else:
            raise RuntimeError('Invalid Action: {}'.format(action))

    def finished(self):
        return (
            (self.i == self.n) and
            (len(self.stack) == 1) and
            (len(self.stack[0][2]) == 1)
        )

    def tree(self):
        if not self.finished():
            raise RuntimeError('Not finished.')
        return self.stack[0][2][0]

    def s_features(self):
        lefts = []
        rights = []

        lefts.append(1)
        if len(self.stack) < 2:
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            rights.append(s1_left)

        if len(self.stack) < 2:
            lefts.append(1)
            rights.append(0)
        else:
            s1_left = self.stack[-2][0] + 1
            lefts.append(s1_left)
            s1_right = self.stack[-2][1] + 1
            rights.append(s1_right)

        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)

    def l_features(self):
        lefts = []
        rights = []

        lefts.append(1)
        if len(self.stack) < 1:
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            rights.append(s0_left)

        if len(self.stack) < 1:
            lefts.append(1)
            rights.append(0)
        else:
            s0_left = self.stack[-1][0] + 1
            lefts.append(s0_left)
            s0_right = self.stack[-1][1] + 1
            rights.append(s0_right)

        lefts.append(self.i + 1)
        rights.append(self.n)

        return tuple(lefts), tuple(rights)

    def s_oracle(self, tree):
        if not self.can_shift():
            return 'comb'
        elif not self.can_combine():
            return 'sh'
        else:
            (left0, right0, _) = self.stack[-1]
            subtree = tree.enclosing(left0, right0, equal=False)
            left = subtree.left_span()
            if left == left0:
                return 'sh'
            else:
                return 'comb'

    def l_oracle(self, tree):
        (left0, right0, _) = self.stack[-1]
        labels, crossing = tree.span_labels(left0, right0)
        labels = labels[::-1]
        if len(labels) == 0:
            return 'none'
        else:
            return 'label-' + '-'.join(labels)

    @staticmethod
    def gold_actions(tree):
        n = len(tree.sentence)
        state = State(n)
        result = []

        for step in range(2 * n - 1):
            if state.can_combine():
                (left0, right0, _) = state.stack[-1]
                (left1, _, _) = state.stack[-2]
                subtree = tree.enclosing(left0, right0, equal=False)
                a, b = subtree.left_span(), subtree.right_span()
                if left1 >= a:
                    result.append('comb')
                    state.combine()
                else:
                    result.append('sh')
                    state.shift()
            else:
                result.append('sh')
                state.shift()

            (left0, right0, _) = state.stack[-1]
            labels, crossing = tree.span_labels(left0, right0)
            labels = labels[::-1]
            if len(labels) == 0:
                result.append('none')
            else:
                result.append('label-' + '-'.join(labels))
                state.label(labels)

        return result

    @staticmethod
    def training_data(tree):
        s_features = []
        l_features = []

        n = len(tree.sentence)
        state = State(n)
        result = []

        for step in range(2 * n - 1):
            if not state.can_combine():
                action = 'sh'
            elif not state.can_shift():
                action = 'comb'
            else:
                action = state.s_oracle(tree)
                features = state.s_features()
                s_features.append((features, action))
            state.take_action(action)

            action = state.l_oracle(tree)
            features = state.l_features()
            l_features.append((features, action))
            state.take_action(action)

        return (s_features, l_features)
