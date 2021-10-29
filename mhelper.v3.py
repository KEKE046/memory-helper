import warnings
if __name__ == '__main__':
    print('Loading...')
    warnings.filterwarnings("ignore")

import pyfiglet
import time
from rich.console import Console
from rich import print as rprint
from sklearn.linear_model import Ridge
from pypinyin import lazy_pinyin
import nagisa
import jieba
import regex
import numpy as np
import gtts
import sounddevice as sd
import librosa
import langdetect
from typing import Any, Callable, Dict, List, Tuple, Union
from pathlib import Path
import random
import copy
import re
import json
import hashlib
from tqdm import tqdm


def simple_digest(s: str):
    return s[:10] + '-' + hashlib.md5(s.encode('utf-8')).hexdigest()


class SerializableObjectMeta(type):
    def __new__(cls, name, bases, attrs):
        annotations = {}
        default_values = {}
        for base in bases[::-1]:
            annotations.update(base.__dict__.get('__annotations__', {}))
            default_values.update(base.__dict__)
        annotations.update(attrs.get('__annotations__', {}))
        default_values.update(attrs)
        default_values = {k: v for k,
                          v in default_values.items() if k in annotations}
        attrs['__serializableobject_fields__'] = annotations
        attrs['__serializableobject_values__'] = default_values
        return type.__new__(cls, name, bases, attrs)


class SerializableObject(metaclass=SerializableObjectMeta):
    '''一个抽象类，继承这个类的，可以调用serialize方法来打散成字典。便于与前端交互和存盘。'''
    def __init__(self, *args:Tuple[Any], **kws:Dict[str, Any]):
        for i, k in enumerate(self.__serializableobject_fields__):
            if i < len(args):
                setattr(self, k, args[i])
            elif k in kws:
                setattr(self, k, kws[k])
            elif k in self.__serializableobject_values__:
                setattr(self, k, self.__serializableobject_values__[k])
            else:
                raise ValueError(f"param {k} not specified")

    def serialize(self):
        ret = {}
        for k, v in self.__serializableobject_fields__.items():
            if type(v) is SerializableObjectMeta:
                ret[k] = self.__dict__[k].serialize()
            else:
                ret[k] = self.__dict__[k]
        return ret

    @classmethod
    def deserialize(cls, data):
        state_dict = {}
        for k, v in cls.__serializableobject_fields__.items():
            if type(v) is SerializableObjectMeta:
                state_dict[k] = v.deserialize(data[k])
            elif k in data:
                state_dict[k] = data[k]
        return cls(**state_dict)

    def __str__(self):
        msg = self.__class__.__name__ + '('
        msg += ', '.join([k + '=' + str(self.__dict__[k])
                         for k in self.__serializableobject_fields__])
        msg += ')'
        return msg

    def __repr__(self):
        return str(self)

    def __eq__(self, rhs):
        if type(rhs.__class__) is not SerializableObjectMeta:
            return False
        for k in self.__serializableobject_fields__:
            if k not in rhs.__serializableobject_fields__:
                return False
            if rhs.__dict__[k] != self.__dict__[k]:
                return False
        return True

    def __ne__(self, rhs):
        return not self.__eq__(rhs)


class MemoryStat(SerializableObject):

    EF: float = 2.5
    interval: int = 0
    upcoming: int = 0

    def decrease_tick(self) -> bool:
        self.upcoming = max(self.upcoming - 1, 0)
        return self.upcoming == 0

    def is_active(self) -> bool:
        return self.upcoming == 0

    def add_stat(self, q: int):
        if q < 3:
            self.interval = 0
            self.upcoming = 0
        self.EF = max(self.EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)), 1.3)
        if self.interval == 0:
            self.interval = 1
        elif self.interval == 1:
            self.interval = 6
        else:
            self.interval = int(round(self.EF * self.interval))
        self.upcoming = self.interval


class Question(SerializableObject):

    title: str = ''
    answer: str = ''
    language: str = ''
    autoplay: bool = False
    memory_stat: MemoryStat = MemoryStat()
    question_id: int = 0
    reconstruct_pattern: str = '**Question** {title}\n{answer}'
    match_method: Union[List[str], None] = None
    match_ignore: Union[List[str], None] = None
    invisible: bool = False

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        if not self.language:
            self.language = langdetect.detect(self.title + self.answer)
        self.title = self.title.strip()
        self.answer = self.answer.strip()

    def get_uid(self):
        return simple_digest(self.title + '#' + self.answer)


class AudioManager:

    def __init__(self, cache_dir: Path):
        cache_dir.mkdir(exist_ok=True, parents=True)
        self.cache_dir = cache_dir
        self.cache_size = sum(f.stat().st_size for f in cache_dir.glob('*.mp3'))
    
    def get_cache_size(self):
        return self.cache_size

    def get_audio(self, title: str, force_download: bool = False, **params: Dict[str, Any]) -> Path:
        name = simple_digest(title)
        path = self.cache_dir.joinpath(f'{name}.mp3')
        if not path.exists() or force_download:
            tts = gtts.gTTS(title.replace('*', ''), **params)
            tts.save(path)
            if not force_download:
                self.cache_size += path.stat().st_size
        return path

    def play_audio(self, data: str, force_download: bool = False, **params: Dict[str, Any]):
        sd.stop()
        path = self.get_audio(data, force_download, **params)
        data, fs = librosa.load(path)
        sd.play(data, fs, blocking=False)


list(jieba.cut('测试结巴分词'))
class MatchManager:

    def __init__(self):
        self.match_method: Dict[str, Callable[[Question, str], Tuple[bool, str]]] = [
            ('full-match', self.full_match),
            ('token-match', self.token_match),
            ('pinyin-match', self.pinyin_match),
            ('char-match', self.char_match),
        ]

    def clean_word(self, s):
        return regex.sub(r'[\p{P}\s]+', ' ', s.strip())

    def split_word_zh(self, s):
        return list(jieba.cut(s.lower()))

    def split_word_ja(self, s):
        return list(nagisa.tagging(s).words)

    def first_pinyin_zh(self, s):
        words = self.split_word_zh(s)
        return [''.join(lazy_pinyin(w, 4)).upper() for w in words]

    def pattern_match(self, patterns: List[str], data: Union[str, List], tag: str = '') -> Tuple[bool, str]:
        '''
            检查patterns中的每一个是否都在data之中出现了
        '''
        match_mask = np.zeros(len(data))
        unmatched = []
        for pat in patterns:
            if pat in data:
                i = data.index(pat)
                if isinstance(data, str):
                    match_mask[i:i+len(pat)] = 1
                else:
                    match_mask[i] = 1
            else:
                unmatched.append(pat)
        msg = f"**{tag}** "
        edge = np.diff(match_mask, prepend=0, append=0)
        if edge[0]:
            msg += "*"
        for c, e in zip(data, edge[1:]):
            msg += c
            if e:
                msg += "*"
        if unmatched:
            msg += '  {' + ' '.join(unmatched) + '}'
        return len(unmatched) == 0, msg

    def clean_first(func) -> Callable[..., Any]:
        '''
            将question.answer和用户的answer除去标点符号和空白字符
            分别作为ground-truth和answer传递给被包装的函数
        '''

        def inner(self, question: Question, answer: str):
            gt = self.clean_word(question.answer)
            answer = self.clean_word(answer)
            return func(self, answer, gt, question)
        return inner

    def then_pattern_match(tag) -> Callable[..., Tuple[bool, str]]:
        '''函数处理完之后，输出 pattern, data， 返回调用pattern_match的结果'''
        def wrapper(func):
            def inner(self, *args, **kws):
                pat, data = func(self, *args, **kws)
                return self.pattern_match(pat, data, tag)
            return inner
        return wrapper

    @clean_first
    def full_match(self, answer: str, gt: str, question: Question):
        if answer == gt:
            return True, "**full-match** success"
        else:
            return False, "**full-match** failed"

    @clean_first
    @then_pattern_match("char-match")
    def char_match(self, answer: str, gt: str, question: Question):
        return list(answer), list(gt)

    @clean_first
    @then_pattern_match("token-match")
    def token_match(self, answer: str, gt: str, question: Question):
        func_name = 'split_word_' + question.language
        if hasattr(self, func_name):
            func = getattr(self, func_name)
            return func(answer), func(gt)
        else:
            return list(answer), list(gt)

    @clean_first
    @then_pattern_match("pinyin-match")
    def pinyin_match(self, answer: str, gt: str, question: Question):
        return answer.split(), ''.join(self.first_pinyin_zh(gt))

    def match_answer(self, question: Question, answer: str):
        all_msg = []
        for name, method in self.match_method:
            need_match = False
            if question.match_method:
                if name in question.match_method:
                    need_match = True
            elif question.match_ignore:
                if name not in question.match_ignore:
                    need_match = True
            else:
                need_match = True
            if need_match:
                mat, msg = method(question, answer)
                if mat:
                    return mat, msg
                else:
                    all_msg.append(msg)
        return False, '\n'.join(all_msg)

    def auto_score(self, question: Question, answer: str, speed_score: float):
        match, match_msg = self.match_answer(question, answer)
        if match:
            return max(speed_score, 3), match_msg
        else:
            return min(speed_score, 3), match_msg


class LLRegresser(SerializableObject):
    max_history: int = 500
    alpha: float = 1
    X: Union[List[List[float]], None] = None
    y: Union[List[List[float]], None] = None

    def add_data(self, X: List[float], y: float):
        if self.X is None:
            self.X = []
        if self.y is None:
            self.y = []
        self.X.append(X)
        self.y.append(y)
        if len(self.X) > self.max_history:
            self.X.pop(0)
            self.y.pop(0)

    def estimate(self, X: List[float]) -> float:
        cur_X = np.asarray(X)[np.newaxis, :]
        cur_X = np.concatenate([cur_X, np.log(cur_X + 1)], axis=1)
        try:
            X = np.asarray(self.X)
            X = np.concatenate([X, np.log(X + 1)], axis=1)
            y = np.asarray(self.y)
            return max(Ridge(alpha=self.alpha).fit(X, y).predict(cur_X)[0], 0.01)
        except:
            return cur_X.sum() * 0.1


class SpeedEstimator:

    def __init__(self, path: Path):
        path.parent.mkdir(exist_ok=True, parents=True)
        self.path = path
        self.estimators: Dict[str, LLRegresser] = {}
        if self.path.exists():
            self.load()

    def save(self):
        with self.path.open('w') as f:
            json.dump({k: v.serialize()
                      for k, v in self.estimators.items()}, f)

    def load(self):
        with self.path.open() as f:
            data = json.load(f)
            self.estimators = {k: LLRegresser.deserialize(
                v) for k, v in data.items()}

    @staticmethod
    def get_feature(question: Question, answer: str):
        def str_feat(s):
            return [
                len(s),
                min(map(len, re.split('\s+', s))),
                len(re.split('\s+', s)),
                min(map(len, s.split('\n'))),
                len(s.split('\n'))
            ]
        return str_feat(question.title) + str_feat(question.answer) + str_feat(answer)

    def add_data(self, question: Question, answer: str, timing: float):
        identifier = question.language
        if identifier not in self.estimators:
            self.estimators[identifier] = LLRegresser()
        X = self.get_feature(question, answer)
        y = timing
        self.estimators[identifier].add_data(X, y)

    def estimate(self, question: Question, answer: str):
        identifier = question.language
        if identifier not in self.estimators:
            self.estimators[identifier] = LLRegresser()
        X = self.get_feature(question, answer)
        return self.estimators[identifier].estimate(X)

    def speed_score(self, question: Question, answer: str, timing: float):
        y = self.estimate(question, answer)
        return min(int(y / timing * 7), 5)


class DataSource:

    default_state = {
        'autoplay': False,
        'question': False,
        'inline': False,
        'language': '',
        'invisible': False,
        'match_method': None,
        'match_ignore': None
    }

    dictation_preset = {
        'autoplay': True,
        'invisible': True
    }

    no_dictation_preset = {
        'autoplay': False,
        'invisible': False
    }

    def __init__(self, markdown_file: Path, database_file: Path):
        markdown_file.parent.mkdir(exist_ok=True, parents=True)
        database_file.parent.mkdir(exist_ok=True, parents=True)
        self.markdown_file = markdown_file
        self.database_file = database_file
        self.q: List[Question] = []
        self.db: Dict[str, Question] = {}

        self._state = copy.copy(self.default_state)
        self.state_stack = []
        self.cmd_pattern = re.compile('(.*?)=(.*?)')
        self.reconstruct_pattern = []
        self.parse_message = []
        self.current_line = 0
        self.question_id = -1
        self.current_question: Union[Question, None] = None
        self.froce_state = {}

    @property
    def state(self):
        self._state.update(self.froce_state)
        return self._state

    @state.setter
    def state(self, value):
        self._state = copy.copy(value)
        self._state.update(self.froce_state)

    def set_force_dictation(self):
        self.froce_state.update(self.dictation_preset)

    def set_force_no_dictation(self):
        self.froce_state.update(self.no_dictation_preset)

    def set_force_voice(self):
        self.froce_state.update(autoplay=True)

    def set_force_no_voice(self):
        self.froce_state.update(autoplay=False)

    def set_config(self, presets=[], forces=[]):
        for preset in presets:
            name = 'set_preset_' + preset.replace('-', '_')
            if hasattr(self, name):
                self.add_message('preset ' + preset)
                getattr(self, name)()
        for force in forces:
            name = 'set_force_' + force.replace('-', '_')
            if hasattr(self, name):
                self.add_message('force ' + force)
                getattr(self, name)()

    def push_stack(self):
        self.state_stack.append(copy.copy(self.state))

    def pop_stack(self):
        self.state = self.state_stack.pop()

    def update_stack(self, *args, **kws):
        self.state.update(*args, **kws)
        self.state.update(self.froce_state)

    def handle_voice(self):
        self.update_stack(autoplay=True)

    def handle_question(self):
        self.update_stack(question=True)

    def handle_inline(self):
        self.update_stack(inline=True)

    def handle_invisible(self):
        self.update_stack(invisible=True)

    def handle_language(self, lang):
        self.update_stack(language=lang)

    def handle_match_method(self, *params):
        self.update_stack(match_method=params)

    def handle_match_ignore(self, *params):
        self.update_stack(match_ignore=params)

    def handle_end_all(self):
        self.state = self.state_stack[0]
        self.state_stack = []

    def handle_end(self):
        self.pop_stack()

    def handle_dictation(self):
        self.update_stack(**self.dictation_preset)

    def add_question(self, title, answer, **kws):
        wrap_params = ['language', 'autoplay',
                       'match_method', 'match_ignore', 'invisible']
        wrap_dict = {k: self.state[k] for k in wrap_params}
        self.question_id += 1
        q = Question(
            title=title,
            answer=answer,
            questoin_id=self.question_id,
            **wrap_dict,
            **kws
        )
        if (uid := q.get_uid()) in self.db:
            q.memory_stat = self.db[uid].memory_stat
        self.current_question = q
        self.q.append(q)
        self.reconstruct_pattern.append(None)
        return q

    def add_message(self, msg):
        self.parse_message.append(
            f'File {self.markdown_file.stem} Line {self.current_line + 1}: {msg}')

    def parse_markdown(self, md):
        self.state.update(self.froce_state)
        ignore_raw_text = False
        for self.current_line, line in enumerate(md.split('\n')):
            line = line.strip()
            if line.startswith('```'):
                ignore_raw_text = not ignore_raw_text
            if ignore_raw_text:
                self.reconstruct_pattern.append(line)
                continue
            inline_command = False
            inline_depth = 0
            ctrl_cmd = ''
            if line.endswith('-->'):
                idx = line.index('<!--')
                line_new, ctrl_cmd = line[:idx], line[idx:]
                for ctrl_part in re.findall('\s*<!--(.*?)-->\s*', ctrl_cmd):
                    part_tokens = ctrl_part.split('&')
                    if any([x.strip() == 'end-all' for x in part_tokens]):
                        self.handle_end_all()
                    elif any([x.strip() == 'end' for x in part_tokens]):
                        self.handle_end()
                    else:
                        self.push_stack()
                        inline_depth += 1
                        for ctrl in part_tokens:
                            ctrl_tokens = ctrl.strip().split('=')
                            cmd = ctrl_tokens[0].replace('-', '_')
                            if len(ctrl_tokens) > 1:
                                params = [x.strip()
                                          for x in ctrl.split('=')[1].split(',')]
                            else:
                                params = []
                            if hasattr(self, 'handle_' + cmd):
                                getattr(self, 'handle_' + cmd)(*params)
                            else:
                                self.add_message(
                                    f'Unknown Control Command {ctrl}')
                if line_new:
                    line = line_new
                    inline_command = True
                else:
                    self.reconstruct_pattern.append(line)
                    continue
            if self.state['inline']:
                tokens = re.split('\s+', line)
                self.add_question(
                    title=tokens[0],
                    answer=' '.join(tokens[1:]),
                    reconstruct_pattern='{title} {answer} %s  ' % ctrl_cmd.strip(
                    )
                )
                self.current_question = None
            elif self.state['question']:
                if mat := re.fullmatch(r'\*\*(.*?)\*\*(.*?)', line):
                    label, title = mat.groups()
                    self.current_question = self.add_question(
                        title, '', reconstruct_pattern='**%s** {title} %s\n{answer}' % (label, ctrl_cmd))
                elif self.current_question is not None:
                    if line:
                        self.current_question.answer += line + '\n'
                elif line != '':
                    self.add_message(f'Ignore line: {line}')
            else:
                self.reconstruct_pattern.append(line)
            if inline_command:
                for _ in range(inline_depth):
                    self.pop_stack()

    def load(self):
        if self.database_file.exists():
            with self.database_file.open() as f:
                for prob in json.load(f):
                    q: Question = Question.deserialize(prob)
                    self.db[q.get_uid()] = q
        with self.markdown_file.open() as f:
            self.parse_markdown(f.read())
        return '\n'.join(self.parse_message)

    def get_questions(self):
        return self.q

    def generate_markdown(self):
        questions = sorted(self.q, key=lambda x: x.question_id)
        ques_out = []
        for q in questions:
            ques_out.append(q.reconstruct_pattern.format(
                title=q.title, answer=q.answer))
        qidx = 0
        reconstructed = []
        for r in self.reconstruct_pattern:
            if r is None:
                reconstructed.append(ques_out[qidx])
                qidx += 1
            else:
                reconstructed.append(r)
        return '\n'.join(reconstructed)

    def save(self) -> Dict[str, Any]:
        with self.markdown_file.open('w') as f:
            f.write(self.generate_markdown())
        with self.database_file.open('w') as f:
            json.dump([x.serialize() for x in self.q],
                      f, ensure_ascii=False, indent=4)


class HistoryStat(SerializableObject):
    
    total_problems: int = 0
    total_failed_problems: int = 0
    total_answering: int = 0
    total_failed_answering: int = 0
    score_distribution: Union[None, List[int]] = None
    max_combo: int = 0
    total_using_time: float = 0

    def __init__(self, *args, **kws):
        super().__init__(*args, **kws)
        if self.score_distribution is None:
            self.score_distribution = [0] * 6 # 0, 1, 2, 3, 4, 5


class StatManager:

    def __init__(self, file: Path):
        file.parent.mkdir(exist_ok=True, parents=True)
        self.file = file
        self.history_stat = HistoryStat()
        self.load()
        

    def load(self):
        self.last_tick = time.time()
        if self.file.exists():
            with self.file.open() as f:
                self.history_stat = HistoryStat.deserialize(json.load(f))

    def save(self):
        with self.file.open('w') as f:
            self.history_stat.total_using_time += time.time() - self.last_tick
            self.last_tick = time.time()
            json.dump(self.history_stat.serialize(), f)

    def __getattr__(self, key):
        if (hs:=self.__dict__.get('history_stat', None)) is not None:
            if key in hs.__dict__: return hs.__dict__[key]
        return self.__dict__[key]

    def __setattr__(self, key, value):
        if (hs:=self.__dict__.get('history_stat', None)) is not None:
            if key in hs.__dict__:
                hs.__dict__[key] = value
        self.__dict__[key] = value

    def get_data(self):
        return self.history_stat.serialize()


class Session:

    def __init__(self, data_srcs: List[DataSource], audio_manager: AudioManager, speed_estimator: SpeedEstimator, match_manager: MatchManager, stat_manager: StatManager):
        self.state = 'loaded'
        self.data_srcs = data_srcs
        self.questions: List[Question] = []
        for src in data_srcs:
            self.questions += src.get_questions()
        self.active_questions: List[Question] = []
        self._decrease_tick()
        self.prob_idx = 0
        self.current_round = []
        self.next_round = []
        self.first_round = True
        self.current_prob: Question = None
        self.failed_probs = []
        self.audio_manager = audio_manager
        self.speed_estimator = speed_estimator
        self.cache_autoplay_audio()
        self.score_func = match_manager.auto_score
        self.total_error = 0
        self.combo = 0
        self.current_timing = None
        self.show_all = False
        self.stat_manager = stat_manager

    def cache_all_audio(self, force_download=False):
        for prob in tqdm(self.questions, desc='cacheing audio'):
            self.audio_manager.get_audio(
                title=prob.title, force_download=force_download, lang=prob.language)

    def cache_autoplay_audio(self, force_download=False):
        for prob in tqdm(self.questions, desc='cacheing audio'):
            if prob.autoplay:
                self.audio_manager.get_audio(
                    title=prob.title, force_download=force_download, lang=prob.language)

    def _decrease_tick(self):
        for q in self.questions:
            q.memory_stat.decrease_tick()
            if q.memory_stat.is_active():
                self.active_questions.append(q)

    def save(self):
        for src in self.data_srcs:
            src.save()
        self.speed_estimator.save()
        self.stat_manager.save()

    def error_msg(self, reason=''):
        return {'state': self.state, "result": 'fail', 'reason': reason}

    def success_msg(self, data):
        data = data or {}
        data['state'] = self.state
        data['result'] = 'success'
        return data

    def when(*state):
        def wrapper(func):
            def inner(self, *args, **kws):
                if self.state not in state:
                    return self.error_msg('can only be called at {}'.format(','.join(state)))
                msg = func(self, *args, **kws)
                return self.success_msg(msg)
            return inner
        return wrapper

    @when("loaded")
    def start(self) -> Dict[str, bool]:
        config = {
            'showall': False,
            'shuffle': True,
        }
        if len(self.active_questions) == 0:
            config['fastforward'] = False
        self.state = 'configuring'
        return config

    @when("configuring")
    def set_config(self, config):
        if 'fastforward' in config and config['fastforward']:
            while not self.active_questions:
                self._decrease_tick()
        if config['showall']:
            self.active_questions = self.questions
            self.show_all = True
        if config['shuffle']:
            random.shuffle(self.active_questions)
        self.state = 'ready'
        self.current_round = self.active_questions
        self.prob_idx = 0

    @when("ready", "round-end")
    def next_prob(self):
        if self.prob_idx < len(self.current_round):
            self.current_prob = self.current_round[self.prob_idx]
            self.state = 'answering'
            data = self.current_prob.serialize()
            data['combo'] = self.combo
            data['total_error'] = self.total_error
            data['round_idx'] = self.prob_idx
            data['round_total'] = len(self.current_round)
            data['round_remain'] = len(self.current_round) - self.prob_idx
            data['total_remain'] = len(
                self.current_round) + len(self.next_round) - self.prob_idx
            data['next_round'] = len(self.next_round)
            return data
        else:
            self.current_round = self.next_round
            random.shuffle(self.current_round)
            self.next_round = []
            self.first_round = False
            if not self.current_round:
                self.state = 'end'
            else:
                self.state = 'round-end'
                self.prob_idx = 0

    @when("answering")
    def score(self, answer: str, timing: float):
        if answer == '':
            return {'score': 0, 'message': 'No input'}
        estimate = self.speed_estimator.estimate(self.current_prob, answer)
        speed_score = self.speed_estimator.speed_score(
            self.current_prob, answer, timing)
        self.current_answer = answer
        self.current_timing = timing
        score, msg = self.score_func(self.current_prob, answer, speed_score)
        msg = f'Timing: {timing:.2f}, Esti. {estimate:.2f}\n' + msg
        return {'score': score, 'message': msg}

    @when("answering")
    def answer(self, q):
        if self.first_round:
            self.stat_manager.total_problems += 1
            if not self.show_all:
                self.current_prob.memory_stat.add_stat(q)
            if q <= 3:
                self.stat_manager.total_failed_problems += 1
                self.failed_probs.append(self.current_prob)
        self.stat_manager.total_answering += 1
        self.stat_manager.score_distribution[q] += 1
        if q >= 4:
            self.combo += 1
        else:
            self.combo = 0
        self.stat_manager.max_combo = max(self.stat_manager.max_combo, self.combo)
        if q <= 3:
            self.stat_manager.total_failed_answering += 1
            self.next_round.append(self.current_prob)
            self.total_error += 1
        if q >= 3 and self.current_timing is not None:
            self.speed_estimator.add_data(
                self.current_prob, self.current_answer, self.current_timing)
            self.current_answer = None
            self.current_timing = None
        if q >= 2:
            self.prob_idx += 1
        self.state = 'ready'
        return {'combo': self.combo}

    @when("round-end")
    def get_failed_last_round(self):
        return {'failed_probs': [prob.serialize() for prob in self.current_round]}

    @when("answering")
    def modify_answer(self, answer):
        self.current_prob.answer = answer

    def get_failed(self):
        return {'failed_probs': [prob.serialize() for prob in self.failed_probs]}

    def get_state(self):
        return {'state': self.state}


class Server:

    def __init__(self, root_path, search_path=None):
        self.root_path = Path(root_path)
        self.search_path = Path(search_path) if search_path else self.root_path
        self.files: List[Path] = list(self.search_path.glob('*.md'))
        self.speed_estimator = SpeedEstimator(
            self.root_path.joinpath('.mhelper', '.speed-estimator.json'))
        self.audio_manager = AudioManager(
            self.root_path.joinpath('.mhelper', '.audio'))
        self.match_manager = MatchManager()
        self.stat_manager = StatManager(self.root_path.joinpath('.mhelper', '.stat.json'))

    def get_file_names(self):
        return [x.stem for x in self.files]

    def new_session(self, indices, presets=[], forces=[]):
        srcs = []
        for idx in indices:
            md_file = self.files[idx]
            db_file = md_file.parent.joinpath(
                '.mhelper', md_file.stem + '.json')
            data_src = DataSource(md_file, db_file)
            data_src.set_config(presets=presets, forces=forces)
            print(data_src.load())
            data_src.add_message
            data_src.save()
            srcs.append(data_src)
        return Session(
            audio_manager=self.audio_manager,
            data_srcs=srcs,
            speed_estimator=self.speed_estimator,
            match_manager=self.match_manager,
            stat_manager=self.stat_manager
        )


class ConsoleFrontend:

    best_console_size = (70, 20)
    logo = '[blue]' + pyfiglet.figlet_format("MemoryHelper") + '[/blue]'
    hint_ver = r'                                                                 [red]v3.0[/red]'
    file_head = r'[green]---------------------------- 请选择文件 ----------------------------[/green]'
    ques_head = r'[green]----------------------------   问 题   ----------------------------[/green]'
    anse_head = r'[green]----------------------------   回 答   ----------------------------[/green]'
    scor_head = r'[green]----------------------------   得 分   ----------------------------[/green]'
    roun_head = r'[green]---------------------------- 一轮结束了 ----------------------------[/green]'
    resu_head = r'[green]----------------------------   答 案   ----------------------------[/green]'
    end_head  = r'[green]----------------------------   结 束   ----------------------------[/green]'
    stat_head = r'[green]----------------------------   统 计   ----------------------------[/green]'
    err_input_int = r'[red]请输入一个整数[/red]'
    err_input_set = r'[red]请输入 {intset} 中的一个数[/red]'
    entry_quit = r'退出'
    entry_select_all = r'[green]都来一遍[/green]'
    entry_stat = r'统计'
    entry_about = r'关于'

    hint_sel_file = r'请选择文件(空格间隔多个文件)：'
    hint_retry = r'[blue]重试[/blue]'
    hint_combos = {
        10: '[blue]' + pyfiglet.figlet_format('Comb 10\n      Good!') + '[/blue]',
        20: '[yellow]' + pyfiglet.figlet_format('      Comb 20\nVery Good!') + '[/yellow]',
        30: '[purple]' + pyfiglet.figlet_format('Comb 30\nPerfect!') + '[/purple]',
        40: '[green]' + pyfiglet.figlet_format('Comb 40\n     Excellent!') + '[/green]',
        50: '[pink]' + pyfiglet.figlet_format('      Comb 50\n You Made It!') + '[/pink]',
        100: '[red]' + pyfiglet.figlet_format('      Comb 100\n Unbelievable!') + '[/red]',
        200: '[red][bold]' + pyfiglet.figlet_format('     Comb 200\n Superman!') + '[/bold][/red]',
    }
    hint_invisible = r'[yellow]-- invisible --[/yellow]'
    hint_round = r'本轮: {round_idx}/{round_total} 下一轮: {next_round} 合计错误: {total_error} Combo: {combo}'
    hint_modify = r'[yellow]请输入修改后的答案[/yellow]'
    hint_esti_score = r'[purple]Score:[/purple] {score}'
    hint_input_score = r'InputScore(0~5):'
    hint_show_fails = r'是否查看上一轮错题[0/1]:'
    hint_show_fails_answer = r'用Ctrl-D查看答案'
    hint_round_end = r'休息一下，开始下一轮'
    hint_force_exit = r'[red][bold]强制退出MemoryHelper[/bold][/red]'
    hint_duration = r'[yellow]在{hour:02d}:{min:02d}内回答了{cnt}个问题[/yellow]'
    hint_max_combo = r'[pink] Max Combo: {combo} [/pink]'
    hint_loading = r'[green]读取中...[/green]'

    config_dictation = r'听写选项([bold]0默认[/bold],1强制听写,2强制不听写)：'
    config_force_voice = r'声音选项([bold]0默认[/bold],1强制有声,2强制无声)：'
    config_showall = r'全部都来一遍([bold]0[/bold]/1): '
    config_shuffle = r'打乱顺序([bold]1[/bold]/0)？'
    config_fastforward = r'没有问题会出现强行要做([bold]0[/bold]/1)？'

    stat_pattern = \
r'''滚过的总问题数：{total_problems}
失败的总问题数：{total_failed_problems}
总回答数：{total_answering}
失败回答数：{total_failed_answering}
历史最大combo：{max_combo}
使用时间：{total_using_time}
分数分布：{score_distribution}
'''
    about_message = \
r'''
                       [purple]KEKE[/purple]的死记硬背辅助软件
           最开始是设计来背政治课的，[blue]思修[bold]军理[/bold]史纲[bold]马原[/bold]离谱性[bold]递增[/bold][/blue]
             后来加入了[bold]中文匹配[/bold]和[bold]文本到语音转换[/bold]，用来听写单词了
                   [yellow]现在的版本可以支持通用的问答记忆[/yellow]
                  非常适合[green]打字远快于手写[/green]的程序猿朋友
        项目主页： https://github.com/KEKE046/memory-helper
                软件遵循 [blue]Apache 2.0[/blue] 协议开源，欢迎提Issue
        但提的Issue可能被KEKE[grey]鸽掉[/grey]，想要新功能可以自己先尝试写一写'''

    def __init__(self, root_path: Path, search_path: Path):
        self.root_path = Path(root_path)
        self.search_path = Path(search_path) if search_path else self.root_path
        self.server = Server(self.root_path, self.search_path)
        self.state = 'welcome'
        self.session = None
        self.console = Console()
        self.start_time = None

    def get_int(self, msg, in_set=[], default=0, multiple=False):
        while True:
            try:
                rprint(msg, end='')
                data = input().strip()
                if not data:
                    return default
                if multiple:
                    data = [int(x) for x in re.split('\s+', data)]
                    if in_set and all([x in in_set for x in data]):
                        return data
                else:
                    data = int(data)
                    if in_set and data in in_set:
                        return data
            except KeyboardInterrupt:
                rprint(self.hint_force_exit)
                exit(0)
            except:
                rprint(self.err_input_int)
            rprint(self.err_input_set.format(
                intset=','.join(map(str, in_set))))

    def statistics(self):
        self.console.clear()
        rprint(self.stat_head)
        stat = self.server.stat_manager.get_data()
        rprint(self.stat_pattern.format(**stat))
        input()
        self.state = 'welcome'
    
    def about(self):
        self.console.clear()
        rprint(self.logo)
        rprint(self.hint_ver)
        rprint()
        rprint(self.about_message)
        input()
        self.state = 'welcome'

    def welcome(self):
        self.console.clear()
        rprint(self.logo)
        rprint(self.hint_ver)
        rprint(self.file_head)
        files = self.server.get_file_names()
        rprint(f'[00] {self.entry_quit}')
        for i, f in enumerate(files):
            rprint(f'[{i + 1:02d}] {f}')
        idx_sel_all = len(files) + 1
        idx_stat = len(files) + 2
        idx_about = len(files) + 3
        rprint(f'[{idx_sel_all:02d}] {self.entry_select_all}')
        rprint(f'[{idx_stat:02d}] {self.entry_stat}')
        rprint(f'[{idx_about:02d}] {self.entry_about}')
        indices = self.get_int(self.hint_sel_file, range(0, idx_about + 1), default=[0], multiple=True)
        if 0 in indices:
            self.state = ''
        elif idx_stat in indices:
            self.state = 'statistics'
        elif idx_about in indices:
            self.state = 'about'
        else:
            if idx_sel_all in indices:
                indices = range(len(files))
            rprint(self.hint_loading)
            dictation = self.get_int(self.config_dictation, [0, 1, 2])
            forces = [[], ["dictation"], ["no-dictation"]][dictation]
            if dictation != 1:
                voice = self.get_int(self.config_force_voice, [0, 1, 2])
                forces += [[], ["voice"], ["no-voice"]][voice]
            self.session = self.server.new_session(
                [x - 1 for x in indices], forces=forces)
            self.state = 'rounding'

    def start(self):
        while self.state:
            getattr(self, self.state)()

    @staticmethod
    def wrap_markdown(s):
        s = re.sub(r'\*\*(.*?)\*\*',
                   lambda x: '[yellow]' + x.group(1) + '[/yellow]', s)
        s = re.sub(
            r'\*(.*?)\*', lambda x: '[blue]' + x.group(1) + '[/blue]', s)
        return s

    def get_long_input(self):
        while True:
            data = []
            try:
                start_time = time.time()
                while True:
                    user_input = input()
                    if user_input == '':
                        break
                    data.append(user_input)
                end_time = time.time()
                data = '\n'.join(data)
                timing = end_time - start_time
                break
            except EOFError:
                rprint('\n' + self.hint_retry, end='')
                input()
            except KeyboardInterrupt:
                rprint(self.hint_force_exit)
                exit(0)
        return data, timing

    def show_combo(self, combo):
        self.console.clear()
        if combo in self.hint_combos:
            rprint(self.hint_combos[combo])
            input()

    def print_question(self, q):
        rprint(self.ques_head)
        if q['invisible']:
            rprint(self.hint_invisible)
        else:
            rprint(self.wrap_markdown(q['title']))
        rprint('')
        rprint(self.anse_head)
        if q['autoplay']:
            self.session.audio_manager.play_audio(q['title'])
        answer, timing = self.get_long_input()
        ret = self.session.score(answer, timing)
        score = ret['score']
        score_msg = ret['message']
        rprint(self.scor_head)
        rprint(self.wrap_markdown(score_msg))
        rprint(self.resu_head)
        if q['invisible']:
            rprint(self.wrap_markdown(q['title']))
        rprint(self.wrap_markdown(q['answer']))
        rprint()
        rprint(self.hint_round.format(**q))
        rprint(self.hint_esti_score.format(score=score))
        score = self.get_int(self.hint_input_score,
                             range(-1, 6), default=score)
        if score == -1:
            rprint(self.hint_modify)
            answer, _ = self.get_long_input()
            self.session.modify_answer(answer)
            q['answer'] = answer
            self.console.clear()
            self.print_question(q)
        else:
            combo = self.session.answer(score)['combo']
            self.max_combo = max(self.max_combo, combo)
            self.show_combo(combo)

    def print_round_end_msg(self):
        rprint(self.roun_head)
        if self.get_int(self.hint_show_fails, [0, 1]):
            failed = self.session.get_failed_last_round()['failed_probs']
            rprint(self.hint_show_fails_answer)
            for prob in failed:
                rprint(self.wrap_markdown(prob['title']))
                try:
                    input()
                except EOFError:
                    rprint(self.wrap_markdown(prob['answer']))
                rprint()
        rprint(self.hint_round_end)
        input()

    def print_end_msg(self):
        rprint(self.end_head)
        duration = self.end_time - self.start_time
        rprint(self.hint_duration.format(hour=int(duration//60//60),
               min=int(duration//60 % 60), cnt=int(self.total_answered)))
        input()
        keys = sorted(list(self.hint_combos.keys()))
        if any([x <= self.max_combo for x in keys]):
            rprint(self.hint_max_combo.format(combo=self.max_combo))
            combo = max([x for x in keys if x <= self.max_combo])
            rprint(self.hint_combos[combo])
            input()

    def rounding(self):
        self.start_time = time.time()
        self.total_answered = 0
        self.max_combo = 0
        config = self.session.start()
        if 'showall' in config:
            config['showall'] = bool(self.get_int(
                self.config_showall, [0, 1], default=0))
        if 'shuffle' in config:
            config['shuffle'] = bool(self.get_int(
                self.config_shuffle, [0, 1], default=1))
        if not config['showall'] and 'fastforward' in config:
            config['fastforward'] = bool(self.get_int(
                self.config_fastforward, [0, 1], default=0))
        self.session.set_config(config)
        while True:
            self.console.clear()
            ret = self.session.next_prob()
            if ret['state'] == 'end':
                break
            elif ret['state'] == 'answering':
                self.print_question(ret)
            elif ret['state'] == 'round-end':
                self.print_round_end_msg()
            self.session.save()
            self.total_answered += 1
        self.end_time = time.time()
        self.print_end_msg()
        self.state = 'welcome'


if __name__ == '__main__':
    import sys
    search = sys.argv[1] if len(sys.argv) > 1 else '.'
    cli = ConsoleFrontend(Path(__file__).parent, search)
    cli.start()
