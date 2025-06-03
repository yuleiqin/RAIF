# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility library of instructions."""

import functools
import random
import re
from typing import List

import immutabledict
import nltk

WORD_LIST = ["western", "sentence", "signal", "dump", "spot", "opposite", "bottom", "potato", "administration", "working", "welcome", "morning", "good", "agency", "primary", "wish", "responsibility", "press", "problem", "president", "steal", "brush", "read", "type", "beat", "trainer", "growth", "lock", "bone", "case", "equal", "comfortable", "region", "replacement", "performance", "mate", "walk", "medicine", "film", "thing", "rock", "tap", "total", "competition", "ease", "south", "establishment", "gather", "parking", "world", "plenty", "breath", "claim", "alcohol", "trade", "dear", "highlight", "street", "matter", "decision", "mess", "agreement", "studio", "coach", "assist", "brain", "wing", "style", "private", "top", "brown", "leg", "buy", "procedure", "method", "speed", "high", "company", "valuable", "pie", "analyst", "session", "pattern", "district", "pleasure", "dinner", "swimming", "joke", "order", "plate", "department", "motor", "cell", "spend", "cabinet", "difference", "power", "examination", "engine", "horse", "dimension", "pay", "toe", "curve", "literature", "bother", "fire", "possibility", "debate", "activity", "passage", "hello", "cycle", "background", "quiet", "author", "effect", "actor", "page", "bicycle", "error", "throat", "attack", "character", "phone", "tea", "increase", "outcome", "file", "specific", "inspector", "internal", "potential", "staff", "building", "employer", "shoe", "hand", "direction", "garden", "purchase", "interview", "study", "recognition", "member", "spiritual", "oven", "sandwich", "weird", "passenger", "particular", "response", "reaction", "size", "variation", "a", "cancel", "candy", "exit", "guest", "condition", "fly", "price", "weakness", "convert", "hotel", "great", "mouth", "mind", "song", "sugar", "suspect", "telephone", "ear", "roof", "paint", "refrigerator", "organization", "jury", "reward", "engineering", "day", "possession", "crew", "bar", "road", "description", "celebration", "score", "mark", "letter", "shower", "suggestion", "sir", "luck", "national", "progress", "hall", "stroke", "theory", "offer", "story", "tax", "definition", "history", "ride", "medium", "opening", "glass", "elevator", "stomach", "question", "ability", "leading", "village", "computer", "city", "grand", "confidence", "candle", "priest", "recommendation", "point", "necessary", "body", "desk", "secret", "horror", "noise", "culture", "warning", "water", "round", "diet", "flower", "bus", "tough", "permission", "week", "prompt", "connection", "abuse", "height", "save", "corner", "border", "stress", "drive", "stop", "rip", "meal", "listen", "confusion", "girlfriend", "living", "relation", "significance", "plan", "creative", "atmosphere", "blame", "invite", "housing", "paper", "drink", "roll", "silver", "drunk", "age", "damage", "smoke", "environment", "pack", "savings", "influence", "tourist", "rain", "post", "sign", "grandmother", "run", "profit", "push", "clerk", "final", "wine", "swim", "pause", "stuff", "singer", "funeral", "average", "source", "scene", "tradition", "personal", "snow", "nobody", "distance", "sort", "sensitive", "animal", "major", "negotiation", "click", "mood", "period", "arrival", "expression", "holiday", "repeat", "dust", "closet", "gold", "bad", "sail", "combination", "clothes", "emphasis", "duty", "black", "step", "school", "jump", "document", "professional", "lip", "chemical", "front", "wake", "while", "inside", "watch", "row", "subject", "penalty", "balance", "possible", "adult", "aside", "sample", "appeal", "wedding", "depth", "king", "award", "wife", "blow", "site", "camp", "music", "safe", "gift", "fault", "guess", "act", "shame", "drama", "capital", "exam", "stupid", "record", "sound", "swing", "novel", "minimum", "ratio", "machine", "shape", "lead", "operation", "salary", "cloud", "affair", "hit", "chapter", "stage", "quantity", "access", "army", "chain", "traffic", "kick", "analysis", "airport", "time", "vacation", "philosophy", "ball", "chest", "thanks", "place", "mountain", "advertising", "red", "past", "rent", "return", "tour", "house", "construction", "net", "native", "war", "figure", "fee", "spray", "user", "dirt", "shot", "task", "stick", "friend", "software", "promotion", "interaction", "surround", "block", "purpose", "practice", "conflict", "routine", "requirement", "bonus", "hole", "state", "junior", "sweet", "catch", "tear", "fold", "wall", "editor", "life", "position", "pound", "respect", "bathroom", "coat", "script", "job", "teach", "birth", "view", "resolve", "theme", "employee", "doubt", "market", "education", "serve", "recover", "tone", "harm", "miss", "union", "understanding", "cow", "river", "association", "concept", "training", "recipe", "relationship", "reserve", "depression", "proof", "hair", "revenue", "independent", "lift", "assignment", "temporary", "amount", "loss", "edge", "track", "check", "rope", "estimate", "pollution", "stable", "message", "delivery", "perspective", "mirror", "assistant", "representative", "witness", "nature", "judge", "fruit", "tip", "devil", "town", "emergency", "upper", "drop", "stay", "human", "neck", "speaker", "network", "sing", "resist", "league", "trip", "signature", "lawyer", "importance", "gas", "choice", "engineer", "success", "part", "external", "worker", "simple", "quarter", "student", "heart", "pass", "spite", "shift", "rough", "lady", "grass", "community", "garage", "youth", "standard", "skirt", "promise", "blind", "television", "disease", "commission", "positive", "energy", "calm", "presence", "tune", "basis", "preference", "head", "common", "cut", "somewhere", "presentation", "current", "thought", "revolution", "effort", "master", "implement", "republic", "floor", "principle", "stranger", "shoulder", "grade", "button", "tennis", "police", "collection", "account", "register", "glove", "divide", "professor", "chair", "priority", "combine", "peace", "extension", "maybe", "evening", "frame", "sister", "wave", "code", "application", "mouse", "match", "counter", "bottle", "half", "cheek", "resolution", "back", "knowledge", "make", "discussion", "screw", "length", "accident", "battle", "dress", "knee", "log", "package", "it", "turn", "hearing", "newspaper", "layer", "wealth", "profile", "imagination", "answer", "weekend", "teacher", "appearance", "meet", "bike", "rise", "belt", "crash", "bowl", "equivalent", "support", "image", "poem", "risk", "excitement", "remote", "secretary", "public", "produce", "plane", "display", "money", "sand", "situation", "punch", "customer", "title", "shake", "mortgage", "option", "number", "pop", "window", "extent", "nothing", "experience", "opinion", "departure", "dance", "indication", "boy", "material", "band", "leader", "sun", "beautiful", "muscle", "farmer", "variety", "fat", "handle", "director", "opportunity", "calendar", "outside", "pace", "bath", "fish", "consequence", "put", "owner", "go", "doctor", "information", "share", "hurt", "protection", "career", "finance", "force", "golf", "garbage", "aspect", "kid", "food", "boot", "milk", "respond", "objective", "reality", "raw", "ring", "mall", "one", "impact", "area", "news", "international", "series", "impress", "mother", "shelter", "strike", "loan", "month", "seat", "anything", "entertainment", "familiar", "clue", "year", "glad", "supermarket", "natural", "god", "cost", "conversation", "tie", "ruin", "comfort", "earth", "storm", "percentage", "assistance", "budget", "strength", "beginning", "sleep", "other", "young", "unit", "fill", "store", "desire", "hide", "value", "cup", "maintenance", "nurse", "function", "tower", "role", "class", "camera", "database", "panic", "nation", "basket", "ice", "art", "spirit", "chart", "exchange", "feedback", "statement", "reputation", "search", "hunt", "exercise", "nasty", "notice", "male", "yard", "annual", "collar", "date", "platform", "plant", "fortune", "passion", "friendship", "spread", "cancer", "ticket", "attitude", "island", "active", "object", "service", "buyer", "bite", "card", "face", "steak", "proposal", "patient", "heat", "rule", "resident", "broad", "politics", "west", "knife", "expert", "girl", "design", "salt", "baseball", "grab", "inspection", "cousin", "couple", "magazine", "cook", "dependent", "security", "chicken", "version", "currency", "ladder", "scheme", "kitchen", "employment", "local", "attention", "manager", "fact", "cover", "sad", "guard", "relative", "county", "rate", "lunch", "program", "initiative", "gear", "bridge", "breast", "talk", "dish", "guarantee", "beer", "vehicle", "reception", "woman", "substance", "copy", "lecture", "advantage", "park", "cold", "death", "mix", "hold", "scale", "tomorrow", "blood", "request", "green", "cookie", "church", "strip", "forever", "beyond", "debt", "tackle", "wash", "following", "feel", "maximum", "sector", "sea", "property", "economics", "menu", "bench", "try", "language", "start", "call", "solid", "address", "income", "foot", "senior", "honey", "few", "mixture", "cash", "grocery", "link", "map", "form", "factor", "pot", "model", "writer", "farm", "winter", "skill", "anywhere", "birthday", "policy", "release", "husband", "lab", "hurry", "mail", "equipment", "sink", "pair", "driver", "consideration", "leather", "skin", "blue", "boat", "sale", "brick", "two", "feed", "square", "dot", "rush", "dream", "location", "afternoon", "manufacturer", "control", "occasion", "trouble", "introduction", "advice", "bet", "eat", "kill", "category", "manner", "office", "estate", "pride", "awareness", "slip", "crack", "client", "nail", "shoot", "membership", "soft", "anybody", "web", "official", "individual", "pizza", "interest", "bag", "spell", "profession", "queen", "deal", "resource", "ship", "guy", "chocolate", "joint", "formal", "upstairs", "car", "resort", "abroad", "dealer", "associate", "finger", "surgery", "comment", "team", "detail", "crazy", "path", "tale", "initial", "arm", "radio", "demand", "single", "draw", "yellow", "contest", "piece", "quote", "pull", "commercial", "shirt", "contribution", "cream", "channel", "suit", "discipline", "instruction", "concert", "speech", "low", "effective", "hang", "scratch", "industry", "breakfast", "lay", "join", "metal", "bedroom", "minute", "product", "rest", "temperature", "many", "give", "argument", "print", "purple", "laugh", "health", "credit", "investment", "sell", "setting", "lesson", "egg", "middle", "marriage", "level", "evidence", "phrase", "love", "self", "benefit", "guidance", "affect", "you", "dad", "anxiety", "special", "boyfriend", "test", "blank", "payment", "soup", "obligation", "reply", "smile", "deep", "complaint", "addition", "review", "box", "towel", "minor", "fun", "soil", "issue", "cigarette", "internet", "gain", "tell", "entry", "spare", "incident", "family", "refuse", "branch", "can", "pen", "grandfather", "constant", "tank", "uncle", "climate", "ground", "volume", "communication", "kind", "poet", "child", "screen", "mine", "quit", "gene", "lack", "charity", "memory", "tooth", "fear", "mention", "marketing", "reveal", "reason", "court", "season", "freedom", "land", "sport", "audience", "classroom", "law", "hook", "win", "carry", "eye", "smell", "distribution", "research", "country", "dare", "hope", "whereas", "stretch", "library", "if", "delay", "college", "plastic", "book", "present", "use", "worry", "champion", "goal", "economy", "march", "election", "reflection", "midnight", "slide", "inflation", "action", "challenge", "guitar", "coast", "apple", "campaign", "field", "jacket", "sense", "way", "visual", "remove", "weather", "trash", "cable", "regret", "buddy", "beach", "historian", "courage", "sympathy", "truck", "tension", "permit", "nose", "bed", "son", "person", "base", "meat", "usual", "air", "meeting", "worth", "game", "independence", "physical", "brief", "play", "raise", "board", "she", "key", "writing", "pick", "command", "party", "yesterday", "spring", "candidate", "physics", "university", "concern", "development", "change", "string", "target", "instance", "room", "bitter", "bird", "football", "normal", "split", "impression", "wood", "long", "meaning", "stock", "cap", "leadership", "media", "ambition", "fishing", "essay", "salad", "repair", "today", "designer", "night", "bank", "drawing", "inevitable", "phase", "vast", "chip", "anger", "switch", "cry", "twist", "personality", "attempt", "storage", "being", "preparation", "bat", "selection", "white", "technology", "contract", "side", "section", "station", "till", "structure", "tongue", "taste", "truth", "difficulty", "group", "limit", "main", "move", "feeling", "light", "example", "mission", "might", "wait", "wheel", "shop", "host", "classic", "alternative", "cause", "agent", "consist", "table", "airline", "text", "pool", "craft", "range", "fuel", "tool", "partner", "load", "entrance", "deposit", "hate", "article", "video", "summer", "feature", "extreme", "mobile", "hospital", "flight", "fall", "pension", "piano", "fail", "result", "rub", "gap", "system", "report", "suck", "ordinary", "wind", "nerve", "ask", "shine", "note", "line", "mom", "perception", "brother", "reference", "bend", "charge", "treat", "trick", "term", "homework", "bake", "bid", "status", "project", "strategy", "orange", "let", "enthusiasm", "parent", "concentrate", "device", "travel", "poetry", "business", "society", "kiss", "end", "vegetable", "employ", "schedule", "hour", "brave", "focus", "process", "movie", "illegal", "general", "coffee", "ad", "highway", "chemistry", "psychology", "hire", "bell", "conference", "relief", "show", "neat", "funny", "weight", "quality", "club", "daughter", "zone", "touch", "tonight", "shock", "burn", "excuse", "name", "survey", "landscape", "advance", "satisfaction", "bread", "disaster", "item", "hat", "prior", "shopping", "visit", "east", "photo", "home", "idea", "father", "comparison", "cat", "pipe", "winner", "count", "lake", "fight", "prize", "foundation", "dog", "keep", "ideal", "fan", "struggle", "peak", "safety", "solution", "hell", "conclusion", "population", "strain", "alarm", "measurement", "second", "train", "race", "due", "insurance", "boss", "tree", "monitor", "sick", "course", "drag", "appointment", "slice", "still", "care", "patience", "rich", "escape", "emotion", "royal", "female", "childhood", "government", "picture", "will", "sock", "big", "gate", "oil", "cross", "pin", "improvement", "championship", "silly", "help", "sky", "pitch", "man", "diamond", "most", "transition", "work", "science", "committee", "moment", "fix", "teaching", "dig", "specialist", "complex", "guide", "people", "dead", "voice", "original", "break", "topic", "data", "degree", "reading", "recording", "bunch", "reach", "judgment", "lie", "regular", "set", "painting", "mode", "list", "player", "bear", "north", "wonder", "carpet", "heavy", "officer", "negative", "clock", "unique", "baby", "pain", "assumption", "disk", "iron", "bill", "drawer", "look", "double", "mistake", "finish", "future", "brilliant", "contact", "math", "rice", "leave", "restaurant", "discount", "sex", "virus", "bit", "trust", "event", "wear", "juice", "failure", "bug", "context", "mud", "whole", "wrap", "intention", "draft", "pressure", "cake", "dark", "explanation", "space", "angle", "word", "efficiency", "management", "habit", "star", "chance", "finding", "transportation", "stand", "criticism", "flow", "door", "injury", "insect", "surprise", "apartment"]  # pylint: disable=line-too-long


WORD_LIST_CN = ["西方的", "句子", "信号", "倾倒", "地点", "相反的", "底部", "土豆", "管理", "工作", "欢迎", "早晨", "好的", "机构", "主要的", "希望", "责任", "按", "问题", "总统", "偷", "刷", "读", "类型", "打", "教练", "增长", "锁", "骨头", "案例", "平等的", "舒适的", "地区", "替换", "表现", "伙伴", "走", "药", "电影", "东西", "岩石", "轻拍", "总计", "竞争", "轻松", "南方", "建立", "聚集", "停车", "世界", "大量", "呼吸", "声称", "酒精", "贸易", "亲爱的", "突出", "街道", "事情", "决定", "混乱", "协议", "工作室", "教练", "帮助", "大脑", "翅膀", "风格", "私人的", "顶部", "棕色", "腿", "买", "程序", "方法", "速度", "高的", "公司", "有价值的", "馅饼", "分析师", "会议", "模式", "区", "愉快", "晚餐", "游泳", "笑话", "订单", "盘子", "部门", "电机", "细胞", "花费", "柜子", "差异", "力量", "检查", "引擎", "马", "维度", "支付", "脚趾", "曲线", "文学", "打扰", "火", "可能性", "辩论", "活动", "通道", "你好", "循环", "背景", "安静的", "作者", "效果", "演员", "页面", "自行车", "错误", "喉咙", "攻击", "角色", "电话", "茶", "增加", "结果", "文件", "具体的", "检查员", "内部的", "潜力", "员工", "建筑", "雇主", "鞋", "手", "方向", "花园", "购买", "面试", "学习", "识别", "成员", "精神的", "烤箱", "三明治", "奇怪的", "乘客", "特别的", "回应", "反应", "大小", "变化", "一个", "取消", "糖果", "出口", "客人", "条件", "飞", "价格", "弱点", "转换", "酒店", "伟大的", "嘴", "心灵", "歌曲", "糖", "怀疑", "电话", "耳朵", "屋顶", "油漆", "冰箱", "组织", "陪审团", "奖励", "工程", "天", "财产", "船员", "酒吧", "道路", "描述", "庆祝", "得分", "标记", "信", "淋浴", "建议", "先生", "运气", "国家的", "进步", "大厅", "中风", "理论", "提供", "故事", "税", "定义", "历史", "骑", "中等", "开口", "玻璃", "电梯", "胃", "问题", "能力", "领先的", "村庄", "电脑", "城市", "宏伟的", "信心", "蜡烛", "牧师", "推荐", "点", "必要的", "身体", "桌子", "秘密", "恐怖", "噪音", "文化", "警告", "水", "圆的", "饮食", "花", "公共汽车", "艰难的", "许可", "周", "提示", "连接", "滥用", "高度", "保存", "角落", "边界", "压力", "驾驶", "停止", "撕裂", "餐", "听", "混乱", "女朋友", "生活", "关系", "重要性", "计划", "创造性的", "气氛", "责备", "邀请", "住房", "纸", "饮料", "卷", "银", "醉酒的", "年龄", "损害", "烟", "环境", "包", "储蓄", "影响", "游客", "雨", "邮政", "标志", "祖母", "跑", "利润", "推动", "职员", "最终", "葡萄酒", "游泳", "暂停", "东西", "歌手", "葬礼", "平均", "来源", "场景", "传统", "个人", "雪", "无人", "距离", "分类", "敏感", "动物", "主要", "谈判", "点击", "情绪", "时期", "到达", "表达", "假期", "重复", "灰尘", "壁橱", "金", "坏", "航行", "组合", "衣服", "强调", "责任", "黑色", "步", "学校", "跳", "文件", "专业", "嘴唇", "化学", "前面", "醒来", "而", "内部", "观看", "排", "主题", "惩罚", "平衡", "可能", "成年人", "旁边", "样本", "上诉", "婚礼", "深度", "国王", "奖", "妻子", "打击", "地点", "营地", "音乐", "安全", "礼物", "错误", "猜测", "行为", "羞愧", "戏剧", "资本", "考试", "愚蠢", "记录", "声音", "摆动", "小说", "最小", "比例", "机器", "形状", "领导", "操作", "薪水", "云", "事务", "打击", "章节", "舞台", "数量", "访问", "军队", "链条", "交通", "踢", "分析", "机场", "时间", "假期", "哲学", "球", "胸部", "感谢", "地方", "山", "广告", "红色", "过去", "租", "返回", "旅游", "房子", "建筑", "网络", "本土", "战争", "数字", "费用", "喷雾", "用户", "污垢", "射击", "任务", "棍", "朋友", "软件", "促销", "互动", "围绕", "块", "目的", "实践", "冲突", "常规", "要求", "奖金", "洞", "状态", "初级", "甜", "抓住", "撕", "折叠", "墙", "编辑", "生活", "位置", "磅", "尊重", "浴室", "外套", "剧本", "工作", "教", "出生", "视图", "解决", "主题", "员工", "怀疑", "市场", "教育", "服务", "恢复", "音调", "伤害", "错过", "工会", "理解", "牛", "河", "协会", "概念", "培训", "食谱", "关系", "保留", "抑郁", "证据", "头发", "收入", "独立", "提升", "任务", "临时", "数量", "损失", "边缘", "轨道", "检查", "绳子", "估计", "污染", "稳定", "信息", "交付", "视角", "镜子", "助理", "代表", "证人", "自然", "法官", "水果", "小费", "魔鬼", "城镇", "紧急", "上层", "掉落", "停留", "人类", "脖子", "发言人", "网络", "唱", "抵抗", "联盟", "旅行", "签名", "律师", "重要性", "气体", "选择", "工程师", "成功", "部分", "外部", "工人", "简单", "四分之一", "学生", "心", "通过", "尽管", "转变", "粗糙", "女士", "草", "社区", "车库", "青年", "标准", "裙子", "承诺", "盲目", "电视", "疾病", "委员会", "积极", "能量", "冷静", "存在", "调音", "基础", "偏好", "头", "共同", "切", "某处", "演示", "当前", "思考", "革命", "努力", "大师", "实施", "共和国", "地板", "原则", "陌生人", "肩膀", "等级", "按钮", "网球", "警察", "收藏", "账户", "注册", "手套", "划分", "教授", "椅子", "优先", "结合", "和平", "扩展", "也许", "晚上", "框架", "姐妹", "波", "代码", "申请", "鼠标", "匹配", "柜台", "瓶子", "一半", "脸颊", "决议", "后面", "知识", "制造", "讨论", "螺丝", "长度", "事故", "战斗", "裙子", "膝盖", "日志", "包裹", "它", "转动", "听证", "报纸", "层", "财富", "个人资料", "想象", "答案", "周末", "老师", "外观", "见面", "自行车", "上升", "腰带", "撞击", "碗", "等价物", "支持", "图像", "诗", "风险", "兴奋", "遥控", "秘书", "公众", "生产", "飞机", "显示", "金钱", "沙", "情况", "打击", "客户", "标题", "摇晃", "抵押贷款", "选项", "数字", "流行", "窗口", "程度", "无", "经验", "意见", "出发", "舞蹈", "指示", "男孩", "材料", "乐队", "领导者", "太阳", "美丽", "肌肉", "农民", "多样性", "肥", "处理", "导演", "机会", "日历", "外面", "步伐", "浴", "鱼", "后果", "放置", "所有者", "去", "医生", "信息", "分享", "伤害", "保护", "职业", "金融", "力量", "高尔夫", "垃圾", "方面", "孩子", "食物", "靴子", "牛奶", "回应", "目标", "现实", "生的", "戒指", "购物中心", "一个", "影响", "区域", "新闻", "国际", "系列", "给人留下深刻印象", "母亲", "庇护所", "罢工", "贷款", "月份", "座位", "任何东西", "娱乐", "熟悉", "线索", "年", "高兴", "超市", "自然", "上帝", "成本", "对话", "领带", "毁灭", "舒适", "地球", "风暴", "百分比", "援助", "预算", "力量", "开始", "睡眠", "其他", "年轻", "单位", "填充", "商店", "欲望", "隐藏", "价值", "杯子", "维护", "护士", "功能", "塔", "角色", "班级", "相机", "数据库", "恐慌", "国家", "篮子", "冰", "艺术", "精神", "图表", "交换", "反馈", "声明", "声誉", "搜索", "狩猎", "锻炼", "恶心", "通知", "男性", "院子", "年度", "领口", "日期", "平台", "植物", "财富", "激情", "友谊", "传播", "癌症", "票", "态度", "岛屿", "活跃", "物体", "服务", "买家", "咬", "卡", "脸", "牛排", "提案", "病人", "热", "规则", "居民", "广泛", "政治", "西方", "刀", "专家", "女孩", "设计", "盐", "棒球", "抓住", "检查", "表兄弟", "情侣", "杂志", "烹饪", "依赖", "安全", "鸡", "版本", "货币", "梯子", "计划", "厨房", "就业", "本地", "注意", "经理", "事实", "封面", "悲伤", "守卫", "亲戚", "县", "比率", "午餐", "程序", "倡议", "齿轮", "桥", "乳房", "谈话", "盘子", "保证", "啤酒", "车辆", "接待", "女人", "物质", "副本", "讲座", "优势", "公园", "寒冷", "死亡", "混合", "保持", "规模", "明天", "血液", "请求", "绿色", "饼干", "教堂", "条", "永远", "超越", "债务", "处理", "洗", "以下", "感觉", "最大", "部门", "海", "财产", "经济学", "菜单", "长椅", "尝试", "语言", "开始", "呼叫", "坚固", "地址", "收入", "脚", "高级", "蜂蜜", "少数", "混合物", "现金", "杂货", "链接", "地图", "形式", "因素", "锅", "模型", "作家", "农场", "冬天", "技能", "任何地方", "生日", "政策", "发布", "丈夫", "实验室", "匆忙", "邮件", "设备", "水槽", "对", "司机", "考虑", "皮革", "皮肤", "蓝色", "船", "销售", "砖", "两个", "喂", "平方", "点", "冲", "梦想", "位置", "下午", "制造商", "控制", "场合", "麻烦", "介绍", "建议", "打赌", "吃", "杀", "类别", "方式", "办公室", "地产", "骄傲", "意识", "滑动", "裂缝", "客户", "钉子", "射击", "会员", "柔软", "任何人", "网络", "官方", "个人", "比萨", "兴趣", "袋", "拼写", "职业", "女王", "交易", "资源", "船", "家伙", "巧克力", "关节", "正式", "楼上", "车", "度假村", "国外", "经销商", "合作", "手指", "手术", "评论", "团队", "细节", "疯狂", "路径", "故事", "初始", "手臂", "收音机", "需求", "单身", "绘制", "黄色", "比赛", "片", "引用", "拉", "商业", "衬衫", "贡献", "奶油", "频道", "西装", "纪律", "指示", "音乐会", "演讲", "低", "有效", "悬挂", "抓伤", "行业", "早餐", "躺", "加入", "金属", "卧室", "分钟", "产品", "休息", "温度", "许多", "给予", "争论", "打印", "紫色", "笑", "健康", "信用", "投资", "出售", "设置", "课程", "蛋", "中间", "婚姻", "水平", "证据", "短语", "爱", "自我", "利益", "指导", "影响", "你", "爸爸", "焦虑", "特别", "男朋友", "测试", "空白", "付款", "汤", "义务", "回复", "微笑", "深", "投诉", "附加", "审查", "盒子", "毛巾", "小", "乐趣", "土壤", "问题", "香烟", "互联网", "获得", "告诉", "入口", "备用", "事件", "家庭", "拒绝", "分支", "可以", "笔", "祖父", "常数", "水箱", "叔叔", "气候", "地面", "体积", "沟通", "善良", "诗人", "孩子", "屏幕", "我的", "退出", "基因", "缺乏", "慈善", "记忆", "牙齿", "恐惧", "提到", "市场营销", "揭示", "原因", "法庭", "季节", "自由", "土地", "运动", "观众", "教室", "法律", "钩", "赢", "携带", "眼睛", "气味", "分配", "研究", "国家", "敢", "希望", "而", "伸展", "图书馆", "如果", "延迟", "大学", "塑料", "书", "现在", "使用", "担心", "冠军", "目标", "经济", "三月", "选举", "反思", "午夜", "滑动", "通货膨胀", "行动", "挑战", "吉他", "海岸", "苹果", "运动", "领域", "夹克", "感觉", "方式", "视觉", "移除", "天气", "垃圾", "电缆", "遗憾", "伙伴", "海滩", "历史学家", "勇气", "同情", "卡车", "紧张", "许可", "鼻子", "床", "儿子", "人", "基础", "肉", "通常", "空气", "会议", "价值", "游戏", "独立", "身体", "简短", "玩", "提高", "董事会", "她", "关键", "写作", "选择", "命令", "派对", "昨天", "春天", "候选人", "物理", "大学", "关注", "发展", "变化", "字符串", "目标", "实例", "房间", "苦", "鸟", "足球", "正常", "分裂", "印象", "木头", "长", "意义", "股票", "帽子", "领导", "媒体", "雄心", "钓鱼", "论文", "沙拉", "修理", "今天", "设计师", "夜晚", "银行", "绘图", "不可避免", "阶段", "广阔", "芯片", "愤怒", "开关", "哭", "扭曲", "个性", "尝试", "存储", "存在", "准备", "蝙蝠", "选择", "白色", "技术", "合同", "侧", "部分", "车站", "直到", "结构", "舌头", "味道", "真相", "困难", "组", "限制", "主要", "移动", "感觉", "光", "例子", "使命", "可能", "等待", "轮子", "商店", "主持", "经典", "替代", "原因", "代理", "组成", "桌子", "航空公司", "文本", "游泳池", "工艺", "范围", "燃料", "工具", "合作伙伴", "负载", "入口", "存款", "仇恨", "文章", "视频", "夏天", "特征", "极端", "移动", "医院", "航班", "秋天", "养老金", "钢琴", "失败", "结果", "擦", "间隙", "系统", "报告", "吸", "普通", "风", "神经", "询问", "闪耀", "笔记", "线", "妈妈", "感知", "兄弟", "参考", "弯曲", "收费", "治疗", "把戏", "术语", "家庭作业", "烘焙", "出价", "状态", "项目", "战略", "橙色", "让", "热情", "父母", "集中", "设备", "旅行", "诗歌", "商业", "社会", "吻", "结束", "蔬菜", "雇佣", "时间表", "小时", "勇敢", "专注", "过程", "电影", "非法", "一般", "咖啡", "广告", "高速公路", "化学", "心理学", "雇佣", "铃", "会议", "救济", "展示", "整洁", "有趣", "重量", "质量", "俱乐部", "女儿", "区域", "触摸", "今晚", "震惊", "燃烧", "借口", "名字", "调查", "风景", "进展", "满意", "面包", "灾难", "项目", "帽子", "优先", "购物", "访问", "东方", "照片", "家", "想法", "父亲", "比较", "猫", "管", "赢家", "计数", "湖", "打斗", "奖品", "基础", "狗", "保持", "理想", "风扇", "挣扎", "巅峰", "安全", "解决方案", "地狱", "结论", "人口", "压力", "警报", "测量", "秒", "火车", "比赛", "到期", "保险", "老板", "树", "监视器", "生病", "课程", "拖拽", "约会", "切片", "仍然", "关心", "耐心", "富有", "逃脱", "情感", "皇家", "女性", "童年", "政府", "图片", "意志", "袜子", "大", "门", "油", "交叉", "别针", "改善", "冠军", "傻", "帮助", "天空", "投球", "男人", "钻石", "大多数", "过渡", "工作", "科学", "委员会", "时刻", "修复", "教学", "挖掘", "专家", "复杂", "指南", "人们", "死亡", "声音", "原创", "打破", "主题", "数据", "学位", "阅读", "录音", "一堆", "达到", "判断", "谎言", "常规", "设置", "绘画", "模式", "列表", "玩家", "熊", "北方", "惊奇", "地毯", "沉重", "官员", "负面", "时钟", "独特", "婴儿", "痛苦", "假设", "磁盘", "铁", "账单", "抽屉", "看", "双重", "错误", "完成", "未来", "辉煌", "联系", "数学", "米", "离开", "餐厅", "折扣", "性别", "病毒", "一点", "信任", "事件", "穿", "果汁", "失败", "虫子", "上下文", "泥", "整体", "包裹", "意图", "草稿", "压力", "蛋糕", "黑暗", "解释", "空间", "角度", "单词", "效率", "管理", "习惯", "明星", "机会", "发现", "交通", "站立", "批评", "流动", "门", "伤害", "昆虫", "惊讶", "公寓"]




# ISO 639-1 codes to language names.
LANGUAGE_CODES = immutabledict.immutabledict({
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "ar": "Arabic",
    "hi": "Hindi",
    "fr": "French",
    "ru": "Russian",
    "de": "German",
    "ja": "Japanese",
    "it": "Italian",
    "bn": "Bengali",
    "uk": "Ukrainian",
    "th": "Thai",
    "ur": "Urdu",
    "ta": "Tamil",
    "te": "Telugu",
    "bg": "Bulgarian",
    "ko": "Korean",
    "pl": "Polish",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "ne": "Nepali",
    "sw": "Swahili",
    "kn": "Kannada",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ml": "Malayalam",
    "fi": "Finnish",
    "zh-cn": "Chinese",
    "zh-tw": "Traditional Chinese",
    })


LANGUAGE_CODES_CN = immutabledict.immutabledict({
    "en": "英文",
    "es": "西班牙文",
    "pt": "葡萄牙文",
    "ar": "阿拉伯文",
    "hi": "印地文",
    "fr": "法文",
    "ru": "俄文",
    "de": "德文",
    "ja": "日文",
    "it": "意大利文",
    "bn": "孟加拉文",
    "uk": "乌克兰文",
    "th": "泰文",
    "ur": "乌尔都文",
    "ta": "泰米尔文",
    "te": "泰卢固文",
    "bg": "保加利亚文",
    "ko": "韩文",
    "pl": "波兰文",
    "he": "希伯来文",
    "fa": "波斯文",
    "vi": "越南文",
    "ne": "尼泊尔文",
    "sw": "斯瓦希里文",
    "kn": "卡纳达文",
    "mr": "马拉地文",
    "gu": "古吉拉特文",
    "pa": "旁遮普文",
    "ml": "马拉雅拉姆文",
    "fi": "芬兰文",
    "zh-cn": "中文",
    "zh-tw": "繁体中文",
    })


_ALPHABETS = "([A-Za-z])"
_PREFIXES = "(Mr|St|Mrs|Ms|Dr)[.]"
_SUFFIXES = "(Inc|Ltd|Jr|Sr|Co)"
_STARTERS = r"(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
_ACRONYMS = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
_WEBSITES = "[.](com|net|org|io|gov|edu|me)"
_DIGITS = "([0-9])"
_MULTIPLE_DOTS = r"\.{2,}"


def split_into_sentences(text):
  """Split the text into sentences.

  Args:
    text: A string that consists of more than or equal to one sentences.

  Returns:
    A list of strings where each string is a sentence.
  """
  text = " " + text + "  "
  text = text.replace("\n", " ")
  text = re.sub(_PREFIXES, "\\1<prd>", text)
  text = re.sub(_WEBSITES, "<prd>\\1", text)
  text = re.sub(_DIGITS + "[.]" + _DIGITS, "\\1<prd>\\2", text)
  text = re.sub(
      _MULTIPLE_DOTS,
      lambda match: "<prd>" * len(match.group(0)) + "<stop>",
      text,
  )
  if "Ph.D" in text:
    text = text.replace("Ph.D.", "Ph<prd>D<prd>")
  text = re.sub(r"\s" + _ALPHABETS + "[.] ", " \\1<prd> ", text)
  text = re.sub(_ACRONYMS + " " + _STARTERS, "\\1<stop> \\2", text)
  text = re.sub(
      _ALPHABETS + "[.]" + _ALPHABETS + "[.]" + _ALPHABETS + "[.]",
      "\\1<prd>\\2<prd>\\3<prd>",
      text,
  )
  text = re.sub(
      _ALPHABETS + "[.]" + _ALPHABETS + "[.]", "\\1<prd>\\2<prd>", text
  )
  text = re.sub(" " + _SUFFIXES + "[.] " + _STARTERS, " \\1<stop> \\2", text)
  text = re.sub(" " + _SUFFIXES + "[.]", " \\1<prd>", text)
  text = re.sub(" " + _ALPHABETS + "[.]", " \\1<prd>", text)
  if "”" in text:
    text = text.replace(".”", "”.")
  if '"' in text:
    text = text.replace('."', '".')
  if "!" in text:
    text = text.replace('!"', '"!')
  if "?" in text:
    text = text.replace('?"', '"?')
  text = text.replace(".", ".<stop>")
  text = text.replace("?", "?<stop>")
  text = text.replace("!", "!<stop>")
  text = text.replace("<prd>", ".")
  sentences = text.split("<stop>")
  sentences = [s.strip() for s in sentences]
  if sentences and not sentences[-1]:
    sentences = sentences[:-1]
  return sentences


def count_words(text):
  """Counts the number of words."""
  tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
  tokens = tokenizer.tokenize(text)
  num_words = len(tokens)
  return num_words


@functools.lru_cache(maxsize=None)
def _get_sentence_tokenizer():
  return nltk.data.load("nltk:tokenizers/punkt/english.pickle")


def count_sentences(text):
  """Count the number of sentences."""
  tokenizer = _get_sentence_tokenizer()
  tokenized_sentences = tokenizer.tokenize(text)
  return len(tokenized_sentences)


def generate_keywords(num_keywords):
  """Randomly generates a few keywords."""
  return random.sample(WORD_LIST, k=num_keywords)
