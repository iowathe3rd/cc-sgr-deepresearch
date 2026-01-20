from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from examples.call_center_mock.contracts import SCHEMA_VERSION

OUTPUT_PATH = Path(__file__).parent / "mock_data" / "call_center_records.json"

SEGMENTS = ["mass_market", "mass_affluent", "small_business", "private_banking"]
CUSTOMER_TIERS = ["bronze", "silver", "gold", "platinum"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
REGIONS = ["almaty", "astana", "shymkent", "aktobe", "karaganda", "atyrau", "kostanay"]

LANGUAGES = ["ru", "kk", "en", "zh", "uz", "ky"]
LANGUAGE_WEIGHTS = [0.6, 0.25, 0.08, 0.03, 0.02, 0.02]

CHANNELS = ["voice", "chat", "email", "messaging"]
CHANNEL_WEIGHTS = [0.7, 0.2, 0.07, 0.03]

SHIFTS = ["morning", "afternoon", "evening", "night"]
SHIFT_HOURS = {
    "morning": (6, 11),
    "afternoon": (12, 16),
    "evening": (17, 21),
    "night": (22, 5),
}

AGENT_LOCATIONS = ["Almaty", "Astana", "Shymkent", "Karaganda", "Remote"]
AGENT_TIERS = ["L1", "L2", "L3"]

BRANCHES = [
    "almaty_center",
    "astana_cbd",
    "shymkent_north",
    "aktobe_market",
    "karaganda_center",
    "atyrau_river",
    "kostanay_hub",
]

QUEUE_PRODUCT_MAP = {
    "cards_support": ["credit_card", "debit_card"],
    "lending_support": ["mortgage", "auto_loan", "consumer_loan"],
    "small_business_support": ["merchant_services", "wire_transfer", "payroll"],
    "retail_support": ["checking", "savings"],
    "fraud_support": ["credit_card", "digital_banking"],
    "wealth_support": ["investment", "private_banking"],
    "payments_support": ["ach_transfer", "swift"],
    "digital_support": ["digital_banking", "mobile_app"],
}

QUEUE_TEAM_MAP = {
    "cards_support": "cards",
    "lending_support": "lending",
    "small_business_support": "business_banking",
    "retail_support": "retail",
    "fraud_support": "fraud",
    "wealth_support": "wealth",
    "payments_support": "payments",
    "digital_support": "digital",
}

ISSUES = {
    "credit_card": [
        {
            "issue_category": "chargeback",
            "issue_subcategory": "merchant_dispute",
            "intent": "file_dispute",
            "tags": ["complaint", "regulatory_risk", "high_value"],
            "topics": ["chargeback", "merchant", "timeline"],
            "root_cause": "merchant dispute and delayed settlement",
            "resolution_code": "chargeback_opened",
            "severity": "high",
            "compliance_flags": ["disclosure_required"],
        },
        {
            "issue_category": "fraud_alert",
            "issue_subcategory": "card_present",
            "intent": "report_fraud",
            "tags": ["fraud_risk", "complaint", "high_value"],
            "topics": ["fraud", "security", "charge"],
            "root_cause": "suspected unauthorized card usage",
            "resolution_code": "fraud_investigation",
            "severity": "high",
            "compliance_flags": ["identity_verification"],
        },
        {
            "issue_category": "credit_limit",
            "issue_subcategory": "increase_request",
            "intent": "increase_limit",
            "tags": ["upsell"],
            "topics": ["limit", "income", "approval"],
            "root_cause": "limit policy review",
            "resolution_code": "limit_review",
            "severity": "medium",
            "compliance_flags": [],
        },
        {
            "issue_category": "travel_notice",
            "issue_subcategory": "international_travel",
            "intent": "set_travel_notice",
            "tags": ["travel_notice"],
            "topics": ["travel", "international", "alerts"],
            "root_cause": "card travel configuration",
            "resolution_code": "travel_notice_set",
            "severity": "low",
            "compliance_flags": [],
        },
    ],
    "debit_card": [
        {
            "issue_category": "lost_card",
            "issue_subcategory": "replacement",
            "intent": "replace_card",
            "tags": ["card_replacement"],
            "topics": ["replacement", "shipping"],
            "root_cause": "card loss",
            "resolution_code": "card_reissue",
            "severity": "medium",
            "compliance_flags": ["identity_verification"],
        }
    ],
    "mortgage": [
        {
            "issue_category": "forbearance",
            "issue_subcategory": "hardship_request",
            "intent": "request_forbearance",
            "tags": ["hardship", "regulatory_risk", "repeat_caller"],
            "topics": ["hardship", "payment", "documentation"],
            "root_cause": "income disruption",
            "resolution_code": "hardship_review",
            "severity": "high",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "auto_loan": [
        {
            "issue_category": "payoff_quote",
            "issue_subcategory": "interest_calculation",
            "intent": "payoff_quote",
            "tags": ["quote_request"],
            "topics": ["payoff", "interest", "schedule"],
            "root_cause": "payoff computation",
            "resolution_code": "quote_generated",
            "severity": "low",
            "compliance_flags": [],
        }
    ],
    "consumer_loan": [
        {
            "issue_category": "payment_extension",
            "issue_subcategory": "grace_period",
            "intent": "extend_payment",
            "tags": ["hardship", "complaint"],
            "topics": ["extension", "grace", "policy"],
            "root_cause": "missed payment due date",
            "resolution_code": "extension_review",
            "severity": "medium",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "merchant_services": [
        {
            "issue_category": "pricing_dispute",
            "issue_subcategory": "interchange_fee",
            "intent": "fee_dispute",
            "tags": ["complaint", "pricing", "repeat_caller"],
            "topics": ["fees", "pricing", "contract"],
            "root_cause": "fee tier confusion",
            "resolution_code": "pricing_review",
            "severity": "high",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "wire_transfer": [
        {
            "issue_category": "payment_failure",
            "issue_subcategory": "incoming_wire_missing",
            "intent": "trace_wire",
            "tags": ["complaint", "high_value", "repeat_caller"],
            "topics": ["wire", "cash_flow", "trace"],
            "root_cause": "beneficiary bank delay",
            "resolution_code": "wire_trace",
            "severity": "high",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "checking": [
        {
            "issue_category": "fee_dispute",
            "issue_subcategory": "overdraft_fee",
            "intent": "fee_reversal",
            "tags": ["complaint", "fee_waiver"],
            "topics": ["overdraft", "fee", "policy"],
            "root_cause": "overdraft policy",
            "resolution_code": "fee_review",
            "severity": "medium",
            "compliance_flags": ["disclosure_required"],
        },
        {
            "issue_category": "atm_issue",
            "issue_subcategory": "cash_withdrawal_error",
            "intent": "atm_dispute",
            "tags": ["complaint", "branch_followup"],
            "topics": ["atm", "cash", "dispute"],
            "root_cause": "atm cash dispense error",
            "resolution_code": "atm_investigation",
            "severity": "medium",
            "compliance_flags": [],
        },
        {
            "issue_category": "service_complaint",
            "issue_subcategory": "agent_behavior",
            "intent": "file_complaint",
            "tags": ["complaint", "service_quality", "repeat_caller"],
            "topics": ["service", "hold_time", "respect"],
            "root_cause": "service quality issue",
            "resolution_code": "qa_review",
            "severity": "high",
            "compliance_flags": [],
        },
    ],
    "investment": [
        {
            "issue_category": "portfolio_review",
            "issue_subcategory": "allocation_question",
            "intent": "portfolio_guidance",
            "tags": ["upsell", "high_value"],
            "topics": ["portfolio", "allocation", "market_outlook"],
            "root_cause": "allocation advisory",
            "resolution_code": "advisor_followup",
            "severity": "medium",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "private_banking": [
        {
            "issue_category": "relationship_management",
            "issue_subcategory": "priority_service",
            "intent": "priority_support",
            "tags": ["high_value", "service_quality"],
            "topics": ["relationship", "service", "priority"],
            "root_cause": "priority handling request",
            "resolution_code": "rm_followup",
            "severity": "high",
            "compliance_flags": [],
        }
    ],
    "ach_transfer": [
        {
            "issue_category": "payment_pending",
            "issue_subcategory": "ach_delay",
            "intent": "ach_status",
            "tags": ["payment_delay", "repeat_caller"],
            "topics": ["ach", "pending", "processing"],
            "root_cause": "processing backlog",
            "resolution_code": "ach_trace",
            "severity": "medium",
            "compliance_flags": [],
        }
    ],
    "swift": [
        {
            "issue_category": "payment_delay",
            "issue_subcategory": "swift_delay",
            "intent": "trace_swift",
            "tags": ["payment_delay", "high_value"],
            "topics": ["swift", "international", "delay"],
            "root_cause": "correspondent bank delay",
            "resolution_code": "swift_trace",
            "severity": "high",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "digital_banking": [
        {
            "issue_category": "login_issue",
            "issue_subcategory": "password_reset",
            "intent": "reset_password",
            "tags": ["digital_deflection"],
            "topics": ["login", "reset"],
            "root_cause": "credential failure",
            "resolution_code": "reset_flow",
            "severity": "low",
            "compliance_flags": [],
        }
    ],
    "mobile_app": [
        {
            "issue_category": "app_bug",
            "issue_subcategory": "crash_on_login",
            "intent": "app_support",
            "tags": ["digital_deflection", "complaint"],
            "topics": ["app", "crash", "update"],
            "root_cause": "app version mismatch",
            "resolution_code": "app_update",
            "severity": "medium",
            "compliance_flags": [],
        }
    ],
    "savings": [
        {
            "issue_category": "interest_rate",
            "issue_subcategory": "rate_change",
            "intent": "rate_explanation",
            "tags": ["complaint"],
            "topics": ["rate", "interest", "policy"],
            "root_cause": "rate policy update",
            "resolution_code": "rate_explained",
            "severity": "low",
            "compliance_flags": ["disclosure_required"],
        }
    ],
    "payroll": [
        {
            "issue_category": "payroll_delay",
            "issue_subcategory": "missing_file",
            "intent": "payroll_trace",
            "tags": ["complaint", "business_critical"],
            "topics": ["payroll", "deadline", "file"],
            "root_cause": "missing payroll file",
            "resolution_code": "payroll_investigation",
            "severity": "high",
            "compliance_flags": [],
        }
    ],
}

TRANSCRIPT_PACKS = {
    "ru": {
        "labels": ("Оператор", "Клиент"),
        "greetings": [
            "Здравствуйте, вы позвонили в службу поддержки банка, меня зовут {agent_name}. Чем могу помочь?",
            "Добрый день, контакт-центр банка, {agent_name} на линии. Расскажите, пожалуйста, в чем вопрос.",
            "Здравствуйте, это банк, служба поддержки. Меня зовут {agent_name}, чем могу быть полезен?",
        ],
        "openings": [
            "Я звоню по вопросу {issue_category} по продукту {product}, сумма {amount} и не понимаю, что произошло.",
            "У меня проблема с {issue_subcategory} и мне нужна помощь прямо сейчас.",
            "Не могу решить вопрос по {product}, уже {repeat_count} раз обращаюсь.",
        ],
        "auth": [
            "Для идентификации назовите, пожалуйста, последние четыре цифры карты или код из SMS.",
            "Сейчас пройдем проверку: назовите дату рождения и код из сообщения.",
            "Подтвердите личность, пожалуйста, через одноразовый код.",
        ],
        "auth_reply": [
            "Код получил, последние цифры {last4}, дата рождения {dob}.",
            "СМС пришло, код {otp}.",
            "Контрольное слово {keyword}, дата рождения {dob}.",
        ],
        "agent_actions": [
            "Я проверяю историю операций и статус обращения.",
            "Давайте уточним детали: когда именно произошло событие и в каком канале?",
            "Я создам кейс {case_id} и передам в профильную команду {team}.",
            "Проверяю настройки и логи по вашему профилю.",
        ],
        "customer_details": [
            "Я видел(а) списание и сразу заблокировал(а) карту, но ответа нет.",
            "Я уже писал(а) в чат и на почту, но ответа пока нет.",
            "Каждый раз меня переводят, и я теряю время.",
            "Мне говорили, что решат за {days} дней, но прошло больше.",
            "Пожалуйста, проверьте, потому что это влияет на мой бизнес.",
        ],
        "frustration": [
            "Я недоволен(на) обслуживанием, ожидание очень долгое.",
            "Это уже не первый раз, я хочу официальную жалобу.",
            "Почему никто не объясняет нормальными словами?",
        ],
        "resolution": [
            "Я запускаю процедуру {resolution_code}, ожидание до {days} дней, подтверждение отправлю.",
            "Мы оформим возврат, сумма будет отражена после проверки.",
            "Я перенаправляю запрос, как только будет ответ, мы сообщим.",
        ],
        "closing_customer": [
            "Хорошо, но пожалуйста, держите меня в курсе.",
            "Спасибо, буду ждать обновлений.",
            "Надеюсь, решите вопрос в этот раз.",
        ],
        "closing_agent": [
            "Спасибо за обращение, мы обязательно вернемся с ответом.",
            "Я отправлю подтверждение на вашу почту, хорошего дня.",
            "Спасибо за звонок, если что-то еще понадобится, мы на связи.",
        ],
        "policy": [
            "Я обязан(а) сообщить, что решение зависит от итогов проверки.",
            "По регламенту мы предоставим ответ в течение установленного срока.",
            "Для безопасности мы не обсуждаем коды подтверждения в открытом виде.",
        ],
        "filler_agent": [
            "Уточните, пожалуйста, когда вы заметили проблему и какие шаги уже предприняли.",
            "Сейчас сверю информацию с системой и историей обращений.",
            "Я вижу ваши прошлые обращения и отмечаю их в кейсе.",
            "Проверяю ограничения и статус в очереди {queue}.",
            "Давайте также проверим альтернативные каналы и уведомления.",
        ],
        "filler_customer": [
            "Я пробовал(а) зайти с другого устройства и обновить приложение.",
            "Было сообщение об ошибке {error_code}, я сделал(а) скриншот.",
            "Я оплачивал(а) покупку, потом увидел(а) двойное списание.",
            "Мне важно получить письменное подтверждение по кейсу.",
            "Я готов(а) прислать документы, если нужно.",
        ],
        "code_switch": ["okay", "mobile app", "email", "refund", "limit"],
    },
    "kk": {
        "labels": ("Оператор", "Клиент"),
        "greetings": [
            "Сәлеметсіз бе, банк колл-орталығы, мен {agent_name}. Қалай көмектесе аламын?",
            "Қайырлы күн, банк қолдау қызметі, {agent_name} тыңдап тұр. Мәселе неде?",
        ],
        "openings": [
            "Менде {product} бойынша {issue_category} мәселесі бар, сома {amount}.",
            "{issue_subcategory} бойынша көмектессеңіз, өтінемін.",
            "Мен бұл мәселе бойынша {repeat_count} рет хабарластым.",
        ],
        "auth": [
            "Тексеру үшін соңғы төрт санды немесе SMS кодты айтыңыз.",
            "Жеке басыңызды растау үшін кодты айтып жіберіңіз.",
        ],
        "auth_reply": [
            "Код келді, {otp}, соңғы төрт саны {last4}.",
            "Туған күнім {dob}, код {otp}.",
        ],
        "agent_actions": [
            "Қазір жүйеден тексеріп жатырмын және іс жүргіземін.",
            "{case_id} нөмірімен кейс ашып, {team} тобына жіберемін.",
            "Мәліметтерді нақтылап алайық, оқиға қашан болды?",
        ],
        "customer_details": [
            "Мен қосымшаны жаңарттым, бірақ қате {error_code} көрсетеді.",
            "Маған бұл өте маңызды, себебі төлемдер кешігіп жатыр.",
            "Мен чатқа да жаздым, бірақ жауап жоқ.",
        ],
        "frustration": [
            "Күту уақыты ұзақ, мені алаңдатады.",
            "Бұл бірінші рет емес, шағым қалдырғым келеді.",
        ],
        "resolution": [
            "{resolution_code} рәсімін бастаймыз, жауап {days} күнде келеді.",
            "Қайтарым рәсімін жібереміз, нәтижесін хабарлаймыз.",
        ],
        "closing_customer": [
            "Жақсы, хабарды күтемін.",
            "Рахмет, бірақ тез шешілсе екен.",
        ],
        "closing_agent": [
            "Рахмет, біз байланыста боламыз.",
            "Күніңіз сәтті өтсін, қосымша сұрақ болса хабарласыңыз.",
        ],
        "policy": [
            "Регламент бойынша жауап белгіленген мерзімде беріледі.",
            "Қауіпсіздік үшін кодтарды ашық айтпаймыз.",
        ],
        "filler_agent": [
            "Алдыңғы өтініштеріңізді көріп тұрмын, барлығын кейске тіркедім.",
            "Қосымша деректер керек болуы мүмкін, қажет болса хабарлаймыз.",
        ],
        "filler_customer": [
            "Менде төлем бойынша қате пайда болды, түсіндіріп беріңіз.",
            "Құжаттарды жіберу керек болса, дайынмын.",
        ],
        "code_switch": ["app", "update", "code", "email"],
    },
    "en": {
        "labels": ("Agent", "Customer"),
        "greetings": [
            "Hello, thank you for calling the bank support line, my name is {agent_name}. How can I help?",
            "Good day, this is the bank contact center. {agent_name} speaking. What can I do for you?",
        ],
        "openings": [
            "I need help with {issue_category} on my {product}; the amount is {amount}.",
            "There is an issue with {issue_subcategory} and I need assistance.",
            "I have already contacted support {repeat_count} times.",
        ],
        "auth": [
            "For verification, please share the last four digits or the SMS code.",
            "Please confirm your identity with the one-time code.",
        ],
        "auth_reply": [
            "The code is {otp} and the last four digits are {last4}.",
            "My date of birth is {dob} and the code is {otp}.",
        ],
        "agent_actions": [
            "I am reviewing the transaction history and your case details now.",
            "I will open case {case_id} and route it to the {team} team.",
            "Let me confirm the timeline and what actions were already taken.",
        ],
        "customer_details": [
            "I tried the mobile app and it shows error {error_code}.",
            "I also wrote an email but have not received a response.",
            "This is impacting my business and I need a clear answer.",
        ],
        "frustration": [
            "The wait time is too long and I am frustrated.",
            "I want to file a formal complaint.",
        ],
        "resolution": [
            "We will start {resolution_code}; the response time is up to {days} days.",
            "I will submit a refund request and keep you updated.",
        ],
        "closing_customer": [
            "Okay, please keep me posted.",
            "Thanks, I will wait for the update.",
        ],
        "closing_agent": [
            "Thank you for calling, we will follow up shortly.",
            "I will send confirmation by email, have a good day.",
        ],
        "policy": [
            "Per policy, the resolution depends on the investigation outcome.",
            "For security reasons, we cannot repeat codes on the call.",
        ],
        "filler_agent": [
            "Could you confirm the device and channel used for the transaction?",
            "I can see your prior contacts and I am adding them to the case.",
        ],
        "filler_customer": [
            "I can provide screenshots or documents if needed.",
            "Please review the prior tickets; they contain the details.",
        ],
        "code_switch": ["ok", "mobile app", "email", "refund"],
    },
    "zh": {
        "labels": ("客服", "客户"),
        "greetings": [
            "您好 这里是 银行 客服中心 我是{agent_name} 请问 需要 什么 帮助？",
            "您好 我是{agent_name} 感谢 来电 有什么 问题 我可以 协助？",
        ],
        "openings": [
            "我的 {product} 出现 {issue_category} 问题 金额 是 {amount} 。",
            "{issue_subcategory} 出了 问题 需要 你们 帮助 。",
        ],
        "auth": [
            "为 验证 身份 请 提供 短信 验证码 或 卡号 后四位 。",
            "请 告知 一次性 验证码 进行 身份 确认 。",
        ],
        "auth_reply": [
            "验证码 是 {otp} 卡号 后四位 是 {last4} 。",
            "我的 生日 是 {dob} 验证码 是 {otp} 。",
        ],
        "agent_actions": [
            "我 正在 查询 系统 记录 并 创建 工单 {case_id} 。",
            "我 会 将 问题 转给 {team} 团队 处理 。",
        ],
        "customer_details": [
            "我 已经 在 App 里 尝试 过 提示 错误 {error_code} 。",
            "我 也 发过 邮件 但 没有 收到 回复 。",
        ],
        "frustration": [
            "等待 时间 太 久 了 我 很 不 满意 。",
            "我 想 提交 正式 投诉 。",
        ],
        "resolution": [
            "我们 将 启动 {resolution_code} 流程 预计 {days} 天内 回复 。",
            "我 会 提交 退款 申请 并 及时 通知 进展 。",
        ],
        "closing_customer": [
            "好的 请 及时 通知 我 。",
            "谢谢 我会 等待 更新 。",
        ],
        "closing_agent": [
            "感谢 来电 我们 会 尽快 跟进 。",
            "我 会 发送 确认 邮件 祝您 一天 愉快 。",
        ],
        "policy": [
            "按照 规定 处理 结果 以 调查 结论 为准 。",
            "出于 安全 考虑 我们 不会 重复 验证码 。",
        ],
        "filler_agent": [
            "请 确认 交易 时间 和 使用 的 渠道 。",
            "我 已 记录 您 之前 的 联系 记录 。",
        ],
        "filler_customer": [
            "如果 需要 我 可以 提供 截图 或 文件 。",
            "我 希望 尽快 解决 这个 问题 。",
        ],
        "code_switch": ["app", "email", "refund"],
    },
    "uz": {
        "labels": ("Operator", "Mijoz"),
        "greetings": [
            "Assalomu alaykum, bank call-markazi, men {agent_name}. Qanday yordam bera olaman?",
            "Salom, bank qo'llab-quvvatlash xizmati, {agent_name} gapiryapti. Muammo nimada?",
        ],
        "openings": [
            "Menda {product} bo'yicha {issue_category} muammosi bor, summa {amount}.",
            "{issue_subcategory} masalasida yordam kerak.",
        ],
        "auth": [
            "Tekshirish uchun SMS kod yoki karta oxirgi to'rt raqamini ayting.",
        ],
        "auth_reply": [
            "Kod {otp}, oxirgi to'rt raqam {last4}.",
        ],
        "agent_actions": [
            "Men tizimdan tekshiraman va {case_id} ishini ochaman.",
            "Masalani {team} jamoasiga yuboraman.",
        ],
        "customer_details": [
            "Ilovada {error_code} xatosi chiqadi.",
            "Oldin ham murojaat qilganman, lekin javob yo'q.",
        ],
        "frustration": [
            "Kutilish juda uzoq.",
            "Rasmiy shikoyat qilmoqchiman.",
        ],
        "resolution": [
            "{resolution_code} jarayoni boshlanadi, javob {days} kun ichida bo'ladi.",
        ],
        "closing_customer": [
            "Rahmat, kutaman.",
        ],
        "closing_agent": [
            "Rahmat, tez orada xabar beramiz.",
        ],
        "policy": [
            "Xavfsizlik uchun kodlarni qayta aytmaymiz.",
        ],
        "filler_agent": [
            "Qo'shimcha ma'lumot kerak bo'lsa, sizga aloqaga chiqamiz.",
        ],
        "filler_customer": [
            "Hujjatlarni yuborishim mumkin.",
        ],
        "code_switch": ["app", "email"],
    },
    "ky": {
        "labels": ("Оператор", "Кардар"),
        "greetings": [
            "Саламатсызбы, банктын колл-борбору, мен {agent_name}. Кандай жардам берем?",
            "Кутман күн, банк колдоо кызматы, {agent_name} угуп турат. Маселе кандай?",
        ],
        "openings": [
            "Менде {product} боюнча {issue_category} маселеси бар, сумма {amount}.",
            "{issue_subcategory} боюнча жардам керек.",
        ],
        "auth": [
            "Текшерүү үчүн SMS кодун же картанын акыркы төрт санын айтыңыз.",
        ],
        "auth_reply": [
            "Код {otp}, акыркы төрт саны {last4}.",
        ],
        "agent_actions": [
            "Мен текшерип жатам жана {case_id} ишин ачам.",
            "Маселени {team} тобуна өткөрөм.",
        ],
        "customer_details": [
            "Колдонмодо {error_code} катасы чыгат.",
            "Мурун да кайрылгам, жооп жок.",
        ],
        "frustration": [
            "Күтүү өтө узак.",
            "Расмий арыз жазгым келет.",
        ],
        "resolution": [
            "{resolution_code} процесси башталат, жооп {days} күн ичинде келет.",
        ],
        "closing_customer": [
            "Рахмат, кабар күтөбүз.",
        ],
        "closing_agent": [
            "Рахмат, тез арада байланышабыз.",
        ],
        "policy": [
            "Коопсуздук үчүн кодду кайталабайбыз.",
        ],
        "filler_agent": [
            "Кошумча маалымат керек болсо, байланышка чыгабыз.",
        ],
        "filler_customer": [
            "Документтерди жөнөтө алам.",
        ],
        "code_switch": ["app", "email"],
    },
}


def _random_date(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, max(delta, 1)))


def _shift_for_datetime(dt: datetime) -> str:
    hour = dt.hour
    for shift, (start, end) in SHIFT_HOURS.items():
        if start <= end and start <= hour <= end:
            return shift
        if start > end and (hour >= start or hour <= end):
            return shift
    return "morning"


def _sentiment_from_tags(tags: list[str], rng: random.Random) -> float:
    base = 0.0
    if {"complaint", "fraud_risk", "pricing", "service_quality", "business_critical"}.intersection(tags):
        base -= 0.45
    if {"hardship", "payment_delay"}.intersection(tags):
        base -= 0.25
    if {"upsell", "digital_deflection", "travel_notice"}.intersection(tags):
        base += 0.25
    jitter = rng.uniform(-0.18, 0.18)
    return max(-1.0, min(1.0, base + jitter))


def _sentiment_label(score: float) -> str:
    if score >= 0.2:
        return "positive"
    if score <= -0.2:
        return "negative"
    return "neutral"


def _resolution_for_sentiment(score: float, rng: random.Random) -> str:
    if score < -0.35 and rng.random() < 0.6:
        return "pending"
    if rng.random() < 0.07:
        return "failed"
    return "resolved"


def _bool_by_probability(rng: random.Random, probability: float) -> bool:
    return rng.random() < probability


def _maybe_missing(rng: random.Random, value, probability: float):
    if rng.random() < probability:
        return None
    return value


def _build_language_mix(rng: random.Random, base_language: str) -> list[str]:
    mix = [base_language]
    if rng.random() < 0.35:
        extra = rng.choice([lang for lang in LANGUAGES if lang != base_language])
        mix.append(extra)
    return mix


def _word_count(text: str) -> int:
    return len(text.split())


def _build_transcript(rng: random.Random, language: str, context: dict) -> str:
    pack = TRANSCRIPT_PACKS[language]
    agent_label, customer_label = pack["labels"]
    target_words = rng.randint(500, 1000)

    lines: list[str] = []
    lines.append(f"{agent_label}: {rng.choice(pack['greetings']).format(**context)}")
    lines.append(f"{customer_label}: {rng.choice(pack['openings']).format(**context)}")
    lines.append(f"{agent_label}: {rng.choice(pack['auth']).format(**context)}")
    lines.append(f"{customer_label}: {rng.choice(pack['auth_reply']).format(**context)}")
    lines.append(f"{customer_label}: {rng.choice(pack['customer_details']).format(**context)}")
    lines.append(f"{agent_label}: {rng.choice(pack['agent_actions']).format(**context)}")

    turns = 0
    max_turns = 140
    while _word_count("\n".join(lines)) < target_words and turns < max_turns:
        lines.append(f"{agent_label}: {rng.choice(pack['filler_agent']).format(**context)}")
        lines.append(f"{customer_label}: {rng.choice(pack['filler_customer']).format(**context)}")
        if rng.random() < 0.25:
            lines.append(f"{agent_label}: {rng.choice(pack['policy']).format(**context)}")
        if rng.random() < 0.2:
            lines.append(f"{customer_label}: {rng.choice(pack['frustration']).format(**context)}")
        if rng.random() < 0.15:
            lines.append(f"{agent_label}: {rng.choice(pack['agent_actions']).format(**context)}")
        if rng.random() < 0.12:
            code_switch = rng.choice(pack["code_switch"])
            lines.append(f"{customer_label}: {code_switch}...")
        turns += 1

    while _word_count("\n".join(lines)) < target_words and turns < max_turns * 2:
        lines.append(f"{agent_label}: {rng.choice(pack['filler_agent']).format(**context)}")
        lines.append(f"{customer_label}: {rng.choice(pack['filler_customer']).format(**context)}")
        turns += 1

    lines.append(f"{agent_label}: {rng.choice(pack['resolution']).format(**context)}")
    lines.append(f"{customer_label}: {rng.choice(pack['closing_customer']).format(**context)}")
    lines.append(f"{agent_label}: {rng.choice(pack['closing_agent']).format(**context)}")

    return "\n".join(lines)


def _financial_impact(rng: random.Random, issue: dict, tags: list[str], duration_sec: int) -> dict:
    severity = issue.get("severity", "low")
    refund = 0.0
    fraud_loss = 0.0
    chargeback = 0.0
    retention = 0.0
    upsell = 0.0

    if "complaint" in tags and rng.random() < 0.4:
        refund = rng.uniform(2000, 25000)
    if "fraud_risk" in tags:
        fraud_loss = rng.uniform(10000, 120000)
    if "chargeback" in issue.get("topics", []):
        chargeback = rng.uniform(5000, 60000)
    if "upsell" in tags and rng.random() < 0.3:
        upsell = rng.uniform(5000, 40000)
    if severity in {"high", "medium"} and rng.random() < 0.25:
        retention = rng.uniform(3000, 30000)

    operational_cost = round((duration_sec / 60.0) * rng.uniform(60, 120), 2)
    revenue_impact = upsell - (refund + fraud_loss + chargeback + retention)

    return {
        "fee_refund_usd": round(refund, 2),
        "fraud_loss_usd": round(fraud_loss, 2),
        "chargeback_amount_usd": round(chargeback, 2),
        "retention_offer_usd": round(retention, 2),
        "upsell_value_usd": round(upsell, 2),
        "revenue_impact_usd": round(revenue_impact, 2),
        "operational_cost_usd": operational_cost,
    }


def generate_records(count: int, start: datetime, end: datetime, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []

    for idx in range(count):
        call_id = f"C{idx + 1:05d}"
        queue = rng.choice(list(QUEUE_PRODUCT_MAP.keys()))
        product = rng.choice(QUEUE_PRODUCT_MAP[queue])
        issue = rng.choice(ISSUES[product])

        call_start = _random_date(rng, start, end)
        duration = rng.randint(240, 2100)
        call_end = call_start + timedelta(seconds=duration)

        segment = rng.choices(SEGMENTS, weights=[0.5, 0.2, 0.2, 0.1], k=1)[0]
        age_band = rng.choice(AGE_BANDS)
        churn_risk = rng.choices(["low", "medium", "high"], weights=[0.5, 0.3, 0.2], k=1)[0]

        tags = list(issue["tags"])
        if rng.random() < 0.22:
            tags.append("repeat_caller")
        if rng.random() < 0.15:
            tags.append("regulatory_risk")
        if rng.random() < 0.1:
            tags.append("complaint")

        sentiment_score = _sentiment_from_tags(tags, rng)
        sentiment_label = _sentiment_label(sentiment_score)

        escalated = _bool_by_probability(
            rng, 0.18 + (0.4 if sentiment_score < -0.25 else 0) + (0.12 if "high_value" in tags else 0)
        )
        fcr = _bool_by_probability(rng, 0.72 if not escalated else 0.18)

        wait_time = rng.randint(10, 220)
        if escalated:
            wait_time += rng.randint(15, 90)
        hold_time = rng.randint(5, 170)
        ivr_time = rng.randint(10, 90)

        abandoned = _bool_by_probability(rng, 0.05 + (0.1 if wait_time > 120 else 0))
        abandon_stage = rng.choice(["ivr", "queue", "agent", "unknown"]) if abandoned else None
        callback_requested = _bool_by_probability(rng, 0.08 + (0.1 if abandoned else 0))
        callback_completed = callback_requested and _bool_by_probability(rng, 0.7)

        resolution_status = _resolution_for_sentiment(sentiment_score, rng)
        resolution_days = rng.randint(0, 10) if resolution_status == "resolved" else None

        nps_last = None if rng.random() < 0.1 else max(0, min(10, int(6 + sentiment_score * 6)))

        language = rng.choices(LANGUAGES, weights=LANGUAGE_WEIGHTS, k=1)[0]
        language_mix = _build_language_mix(rng, language)
        channel = rng.choices(CHANNELS, weights=CHANNEL_WEIGHTS, k=1)[0]
        priority = "vip" if segment == "private_banking" or "high_value" in tags else rng.choice(["low", "medium", "high"])
        shift = _shift_for_datetime(call_start)

        agent_quality = rng.randint(72, 97)
        empathy_score = round(rng.uniform(0.55, 0.95) if sentiment_score > -0.2 else rng.uniform(0.35, 0.75), 2)
        script_adherence = round(rng.uniform(0.6, 0.95), 2)
        compliance_score = round(rng.uniform(0.65, 0.98), 2)
        resolution_confidence = round(rng.uniform(0.45, 0.95) if resolution_status == "resolved" else rng.uniform(0.2, 0.6), 2)

        transcript_context = {
            "agent_name": f"Agent {rng.randint(10, 99)}",
            "issue_category": issue["issue_category"],
            "issue_subcategory": issue["issue_subcategory"],
            "product": product,
            "amount": f"{rng.randint(5000, 180000)} KZT",
            "repeat_count": rng.randint(1, 4),
            "last4": rng.randint(1000, 9999),
            "otp": rng.randint(100000, 999999),
            "dob": rng.choice(["1984-07-12", "1991-03-05", "1976-11-21", "1998-02-09"]),
            "keyword": rng.choice(["alpha", "delta", "sana", "tumar", "orna"]),
            "case_id": f"CS-{rng.randint(100000, 999999)}",
            "team": QUEUE_TEAM_MAP[queue],
            "days": rng.randint(2, 10),
            "error_code": rng.choice(["E42", "E87", "AUTH12", "NET5", "APP09"]),
            "queue": queue,
            "resolution_code": issue["resolution_code"],
        }

        transcript = _build_transcript(rng, language, transcript_context)
        if rng.random() < 0.04:
            transcript = ""
        elif rng.random() < 0.04:
            transcript = transcript[: rng.randint(400, 1200)] + " ... [truncated]"

        key_phrases = issue["topics"] + [issue["issue_category"], issue["issue_subcategory"], queue]

        toxicity_score = round(
            max(0.0, min(1.0, 0.15 + (-sentiment_score * 0.5) + (0.2 if "complaint" in tags else 0))),
            2,
        )

        compliance_flags = list(issue.get("compliance_flags", []))
        if not fcr and rng.random() < 0.15:
            compliance_flags.append("no_first_contact_resolution")
        if not transcript and rng.random() < 0.4:
            compliance_flags.append("missing_transcript")

        quality_flags = []
        if script_adherence < 0.7:
            quality_flags.append("low_script_adherence")
        if empathy_score < 0.5:
            quality_flags.append("low_empathy")

        financials = _financial_impact(rng, issue, tags, duration)

        record = {
            "call_id": call_id,
            "call_start": call_start.isoformat().replace("+00:00", "Z"),
            "call_end": call_end.isoformat().replace("+00:00", "Z"),
            "duration_sec": duration,
            "shift": shift,
            "priority": priority,
            "is_outbound": _bool_by_probability(rng, 0.07),
            "queue": queue,
            "channel": channel,
            "language": language,
            "region": rng.choice(REGIONS),
            "branch": rng.choice(BRANCHES),
            "agent": {
                "agent_id": f"A{rng.randint(100, 799)}",
                "name": f"Agent {rng.randint(10, 99)}",
                "team": QUEUE_TEAM_MAP[queue],
                "tenure_months": rng.randint(3, 84),
                "location": rng.choice(AGENT_LOCATIONS),
                "quality_score": agent_quality,
                "tier": rng.choices(AGENT_TIERS, weights=[0.6, 0.3, 0.1], k=1)[0],
                "manager_id": f"M{rng.randint(10, 99)}",
                "languages": rng.sample(LANGUAGES, k=rng.randint(1, 2)),
            },
            "caller": {
                "customer_id": f"U{rng.randint(10000, 99999)}",
                "segment": segment,
                "tenure_years": round(rng.uniform(0.5, 15.0), 1),
                "age_band": age_band,
                "products": [product] if rng.random() < 0.78 else [product, "savings"],
                "risk_score": round(rng.uniform(0.05, 0.85), 2),
                "churn_risk": churn_risk,
                "nps_last": nps_last,
                "delinquency_days": rng.choice([0, 0, 0, 5, 10, 15, 30, 60]),
                "digital_usage_score": round(rng.uniform(0.2, 0.95), 2),
                "avg_monthly_balance_usd": round(rng.uniform(500, 350000), 2),
                "income_band": rng.choice(["<30k", "30k-50k", "40k-70k", "70k-100k", "100k-150k", "150k-200k", ">200k"]),
                "lifetime_value_usd": round(rng.uniform(1500, 500000), 2),
                "primary_language": language,
                "customer_tier": rng.choices(CUSTOMER_TIERS, weights=[0.45, 0.3, 0.2, 0.05], k=1)[0],
                "complaints_90d": rng.randint(0, 4),
                "recent_calls_30d": rng.randint(1, 6),
                "kyc_status": rng.choices(["verified", "partial", "failed"], weights=[0.9, 0.08, 0.02], k=1)[0],
            },
            "interaction": {
                "wait_time_sec": _maybe_missing(rng, wait_time, 0.05),
                "hold_time_sec": _maybe_missing(rng, hold_time, 0.05),
                "ivr_time_sec": _maybe_missing(rng, ivr_time, 0.06),
                "transfer_count": rng.randint(0, 3),
                "after_call_work_sec": _maybe_missing(rng, rng.randint(40, 260), 0.08),
                "first_call_resolution": fcr,
                "escalated": escalated,
                "escalation_reason": rng.choice(
                    ["policy_dispute", "hardship_review", "pricing_dispute", "fraud_investigation", None]
                )
                if escalated
                else None,
                "abandoned": abandoned,
                "abandon_stage": abandon_stage,
                "callback_requested": callback_requested,
                "callback_completed": callback_completed,
                "auth_method": rng.choice(["otp_sms", "otp_app", "knowledge_based", "voice_biometrics"]),
                "auth_passed": _bool_by_probability(rng, 0.95 if not abandoned else 0.88),
            },
            "case": {
                "case_id": f"CS-{rng.randint(100000, 999999)}",
                "product": product,
                "issue_category": issue["issue_category"],
                "issue_subcategory": issue["issue_subcategory"],
                "resolution": resolution_status,
                "resolution_time_days": resolution_days,
                "case_status": "closed" if resolution_status == "resolved" else "pending",
                "reopen_count": rng.randint(0, 2) if "repeat_caller" in tags else 0,
                "sla_breach": wait_time > 120 or (resolution_days or 0) > 7,
                "root_cause": issue["root_cause"],
                "resolution_code": issue["resolution_code"],
                "follow_up_required": resolution_status != "resolved",
            },
            "nlp": {
                "sentiment_score": round(sentiment_score, 3),
                "sentiment_label": sentiment_label,
                "emotion_peaks": rng.sample(["frustration", "concern", "relief", "anger", "confidence", "anxiety"], k=2),
                "topics": issue["topics"],
                "tags": tags,
                "summary": f"Caller contacted {queue.replace('_', ' ')} about {issue['issue_category']}.",
                "call_intent": issue["intent"],
                "resolution_summary": "Resolved or follow-up scheduled.",
                "transcript": transcript,
                "language_mix": language_mix,
                "toxicity_score": toxicity_score,
                "compliance_flags": compliance_flags,
                "key_phrases": key_phrases,
                "named_entities": [product, queue, issue["issue_category"]],
            },
            "quality": {
                "silence_ratio": round(rng.uniform(0.03, 0.22), 2),
                "overlap_ratio": round(rng.uniform(0.01, 0.1), 2),
                "speech_rate_wpm": rng.randint(120, 190),
                "script_adherence": script_adherence,
                "empathy_score": empathy_score,
                "resolution_confidence": resolution_confidence,
                "csat_pred": round(max(0.0, min(10.0, 5.5 + sentiment_score * 3.5)), 1),
                "compliance_score": compliance_score,
                "qa_flags": quality_flags,
            },
            "financials": financials,
        }

        if rng.random() < 0.03:
            record["call_end"] = (call_start - timedelta(seconds=rng.randint(1, 120))).isoformat().replace(
                "+00:00", "Z"
            )

        records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock call center dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="2024-06-01")
    parser.add_argument("--end", type=str, default="2024-09-30")
    parser.add_argument("--count", type=int, default=300)
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "source_system": "mock_call_center",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": generate_records(args.count, start_dt, end_dt, args.seed),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {args.count} records to {args.output}")


if __name__ == "__main__":
    main()
