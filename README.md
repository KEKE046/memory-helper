# Memory Helper

**KeKe** 和小伙伴们为了完成 *公共基础日语课* 做了一个 *死记硬背* 软件。

[TOC]

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动

```bash
python mhelper.v3.py # 工作目录为当前目录
python mhelper.v3.py workfolder # 指定工作目录
```

运行 `python mhelper.v3.py` 并选择 `README` 文件来查看下面的Demo

## Hint

* 在回答问题的时候，输入一个空行表示回答结束
* `Ctrl-D` 可以重新输入
* `Ctrl-C` 强制退出
* 输入`-1`分，可以修改问题的答案（问题打错的时候很有用）
* `Timing: xxx, Esti. xxx` 中的 `Esti.` 是通过岭回归得到的估计时间，用于评价得分
* **非常欢迎你制作一些mhelper的文件上传到本repo！大家一起共享**


## 分数含义

由于估计的分数可能不准，每个问题回答完后，可能需要用户自己为自己评分。

你也可以回车跳过，采用程序估计的分数。

* `5` 记忆清晰
* `4` 想一会能想起
* `3` 想很长时间才能想起
* `2` 没记全
* `1` 几乎没记忆
* `0` 完全不知道

## 这是Demo


Memory Helper 被设计来可以直接读取 Markdown，你可以运行 `python mhelper.v3.py` 并选择 `README` 文件来查看下面的Demo。

当你平时用markdown来做笔记的时候，可以在笔记里加入一些问题，然后就可以配合mhelper来复习。

**问答题的demo**

<!-- question & voice & language=zh -->
**Q1** 四大名著是什么？（你可以用简拼回答，如 *聊斋* 可以用 *LZ* 来回答） 
三国演义 红楼梦 水浒传 西游记

**Q2** 四大名著之首是什么？ 
红楼梦

**Q3** 四大名著之首是什么？ 这道题你必须用 **全名** 回答，不能用简拼 <!-- match-method=full-match -->
红楼梦

<!-- end-all -->

**记单词的demo** （要把声音打开）

<!-- inline & language=en -->
<!-- dictation -->
welcome welcome，欢迎   
memory memory，记忆   
helper helper，助手   
<!-- end-all -->

## mhelper文件格式

### question

`<!--question-->` 标记一段问题的开始，问题的格式为：
```markdown
<!--question-->
**Label1** question title
question answer
question answer
...
**Q2** question title
question answer
question answer
<!--end-->
```
其中Label1, Q2并不会显示出来，仅仅用于标记一个问题开始。

### inline

`<!--inline-->` 标记单行问题，问题和答案在同一行，中间用空格隔开
```markdown
<!--inline-->
Q1 A1
Q2 A21 A22
<!--end-->
```
有多个空格时，第一个空格之前的被认为是标题，之后的认为是答案。

### voice, language

`<!--voice-->` 标记问题需要阅读，一般与其它标签一起使用
`<!--language=xx-->`用于标记问题的语言，遵循ISO-639-1规范，[语言列表](https://zh.wikipedia.org/wiki/ISO_639-1%E4%BB%A3%E7%A0%81%E8%A1%A8)。
```markdown
<!--inline-->
<!--voice-->
<!--language=en-->
hello world
world hello
<!--end-->
<!--end-->
<!--end-->
```
这样mhelper在显示问题的时候会同时阅读问题的标题。

mhelper会将音频缓存到`.mhelper/.audio`目录下。如需要清除缓存，请去目录下将其删除。

### invisible

`<!--invisible-->` 标记问题标题不可见，一般用于听写。

### 如何听写单词

你可以使用dictation标签，end太多可以用end-all替代。
```markdown
<!--dictation-->
<!--inline-->
<!--language=ja-->

<!--end-all-->
```

也可以结合voice, invisible可以实现单词听写：
```markdown
<!-- inline & voice & language=ja & invisible -->
<!-- match-method=full-match,token-match -->
ここ ここ
紅葉 もみじ
有名 ゆうめい
<!-- end -->
<!-- end -->
```

### 匹配方法

`<!--match-method=xxxxxx-->` 用于指定自动评分方法，可用的方法有：
* `full-match`: 只有用户输入与答案完全匹配才算对
* `char-match`: 用户输入的每一个字符都在答案中出现就算对
* `token-match`：用户输入的每一个词都在答案中出现就算对
* `pinyin-match`：用户输入的大写字母都是答案中某些词首拼就算对

如果没有指定，会将所有评分方法都跑一片，如果有一个对就算对

`<!--ingore-match=xxx-->` 用于指定忽略的方法

你可以用逗号分隔多个方法：

```markdown
<!-- match-method=full-match,token-match -->
```

#### Mhelper的自动评分

MHelper的评分分成两个部分，第一部分是自动匹配，第二部分是按速度给分。

**自动匹配** 就是按照上面的匹配方法，得到用户输入与答案是否匹配。

**速度给分** 将用户过去的输入进行岭回归（一种线性回归），得到的时间为$T$，与当前输入所花时间$t$做比较，估计得分为：
$$\text{speed-score} = \text{min}\left(\frac{7T}{t},\ 5\right)$$

**自动给分** 最后的得分为：

$$
\text{score} = \begin{cases}
\max(3,\ \text{speed-score}), &\text{自动匹配成功}\\
\min(3,\ \text{speed-score}), &\text{自动匹配失败}
\end{cases}
$$

### 行内标记

在单行的末尾也可以添加标记

```markdown
<!-- inline & language=ja -->
野菜 やさい
果物 くだもの
バナナ ばなな
father father <!-- language=en & dictation -->
<!-- end-all -->
```
可以指定banana的语言是en
