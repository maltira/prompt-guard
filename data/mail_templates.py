"""компоненты реалистичных писем для носителей: заголовки, цитирование, пересылка, подписи, уведомления, рассылки

build(rng, lang, body) возвращает абзацы нейтрального письма или документа
"""
import random

NAMES = {
    "en": ["Alex Morgan", "Maria Lopez", "John Carter", "Sophie Bell", "Daniel Reed", "Emma Stone", "Oliver Grant", "Nina Brooks",
           "Peter Hayes", "Laura Kim", "Victor Shaw", "Grace Miller"],
    "ru": ["Алексей Морозов", "Мария Лебедева", "Иван Карпов", "София Белова", "Даниил Орлов", "Елена Громова", "Олег Федоров",
           "Нина Соколова", "Пётр Алексеев", "Лариса Ким", "Виктор Шаров", "Ольга Миронова"],
}
DOMAINS = ["northwind.example", "bluefield.example", "orion-retail.example", "delta-parts.example", "company.example", "mail.example"]
COMPANIES = {
    "en": ["Northwind Logistics", "Bluefield Consulting", "Orion Retail", "Delta Parts", "Summit Software"],
    "ru": ["ООО «Север-Снаб»", "ООО «Вектор»", "АО «Ритм»", "ООО «Лидер Сервис»", "ООО «Горизонт»"],
}
TITLES = {
    "en": ["Project Manager", "Account Executive", "Finance Analyst", "HR Specialist", "Support Engineer", "Office Manager", "Head of Sales"],
    "ru": ["Менеджер проекта", "Главный бухгалтер", "Специалист по персоналу", "Инженер поддержки", "Руководитель отдела продаж", "Офис-менеджер"],
}
SUBJECTS = {
    "en": ["Quarterly sync", "Invoice question", "Delivery update", "Team lunch", "Contract draft", "Travel plans", "Support request",
           "Product feedback", "Re: meeting notes", "Fwd: updated schedule", "Action required: timesheets", "Question about the proposal",
           "Follow-up from Tuesday", "Budget review", "Onboarding checklist", "Access request", "Weekly report", "Vendor contract renewal",
           "Interview availability", "Server maintenance window", "Customer escalation", "Holiday schedule"],
    "ru": ["Квартальная встреча", "Вопрос по счёту", "Статус доставки", "Обед команды", "Проект договора", "Планы поездки",
           "Обращение в поддержку", "Отзыв о продукте", "Re: итоги встречи", "Fwd: обновлённый график", "Нужно действие: табели",
           "Вопрос по предложению", "Продолжение вторничного разговора", "Обзор бюджета", "Чек-лист для нового сотрудника",
           "Запрос доступа", "Еженедельный отчёт", "Продление договора с поставщиком", "Время для собеседования",
           "Окно обслуживания сервера", "Эскалация от клиента", "График на праздники"],
}
DISCLAIMER = {
    "en": ["This message is confidential and intended only for the addressee. If you received it by mistake, please delete it and notify the sender.",
           "Please consider the environment before printing this email."],
    "ru": ["Это сообщение конфиденциально и предназначено только адресату. Если вы получили его по ошибке, удалите его и сообщите отправителю.",
           "Пожалуйста, подумайте об экологии, прежде чем распечатывать это письмо."],
}
GREET = {
    "en": ["Hi {n},", "Hello {n},", "Dear {n},", "Good morning {n},", "Hey {n},", "{n},"],
    "ru": ["Привет, {n}!", "Здравствуйте, {n}!", "Добрый день, {n}!", "{n}, добрый день!", "Уважаемая команда,", "Коллеги, здравствуйте!"],
}
CLOSE = {
    "en": ["Best regards,", "Thanks,", "Cheers,", "Kind regards,", "Regards,"],
    "ru": ["С уважением,", "Спасибо,", "Всего доброго,", "До связи,", "Хорошего дня!"],
}
EXTRA_BODY = {
    "en": [
        "Could you review the attached presentation before the call?", "I've updated the spreadsheet with the latest figures.",
        "Please confirm whether the new deadline works for your team.", "The client asked for a short summary of last month's results.",
        "We are still waiting for the signed copy of the contract.", "The meeting room on the fourth floor is booked until noon.",
        "Thanks for flagging the issue; our developers are looking into it.", "Let me know if you need any additional information from our side.",
        "The refund has been processed and should appear within five business days.", "Our office will be closed on Monday for the public holiday.",
        "I'm forwarding the thread so you have the full context.", "Please find the agenda for tomorrow's workshop below.",
        "The warranty for this model covers parts and labor for two years.", "We have shortlisted three candidates for the second interview round.",
        "The invoice total includes shipping and VAT.", "I can join the call at 3 pm, but I have to leave by 3:45.",
        "The test results look stable, but we want to run one more check.", "Kindly send me the updated price list when you have a moment.",
        "Training materials for the new CRM are available on the intranet.", "The delivery window is between 10 am and 2 pm.",
        "We noticed a small mismatch in the quantities and are checking it with the warehouse.", "Happy to discuss the proposal in more detail this week.",
        "A reminder that the expense reports are due on Friday.", "The building will have no hot water on Thursday morning due to repairs.",
        "Please bring your ID to the front desk on the first day.", "Our support team is available Monday to Friday from 9 to 18.",
        "The new version of the app will be released next Tuesday.", "I wanted to follow up on my previous message about the contract terms.",
        "We appreciate your patience while we resolve this.", "The photos from the site visit are attached.",
        "Can we move the one-on-one to Wednesday morning?", "The auditors will visit our office on the 12th.",
        "Your subscription renews automatically on the first of the month.", "The conference call details are in the calendar invitation.",
        "The prototype passed the first round of testing.", "We are updating our privacy policy; the changes take effect next month.",
        "Please keep receipts for all travel expenses.", "The seminar will be recorded and shared afterwards.",
        "Thank you for choosing our service.", "The report has been shared with the whole team.",
    ],
    "ru": [
        "Посмотрите, пожалуйста, приложенную презентацию до созвона.", "Я обновил таблицу, в ней свежие цифры.",
        "Подтвердите, подходит ли вашей команде новый срок.", "Клиент попросил краткое резюме результатов прошлого месяца.",
        "Мы всё ещё ждём подписанный экземпляр договора.", "Переговорная на четвёртом этаже занята до полудня.",
        "Спасибо, что сообщили о проблеме, разработчики уже разбираются.", "Если нужна дополнительная информация с нашей стороны, дайте знать.",
        "Возврат средств оформлен и поступит в течение пяти рабочих дней.", "В понедельник офис не работает из-за праздника.",
        "Пересылаю цепочку, чтобы у вас был полный контекст.", "Повестка завтрашнего семинара ниже.",
        "Гарантия на эту модель покрывает детали и работу в течение двух лет.", "Мы отобрали трёх кандидатов на второй этап собеседования.",
        "В итоговую сумму счёта входят доставка и НДС.", "Я смогу подключиться к звонку в 15:00, но к 15:45 мне нужно уйти.",
        "Результаты тестов выглядят стабильно, но мы хотим провести ещё одну проверку.", "Пришлите, пожалуйста, обновлённый прайс-лист, когда будет время.",
        "Материалы по обучению новой CRM лежат на внутреннем портале.", "Окно доставки с 10:00 до 14:00.",
        "Мы заметили небольшое расхождение в количестве и сверяем его со складом.", "Буду рад подробнее обсудить предложение на этой неделе.",
        "Напоминаем, что авансовые отчёты нужно сдать до пятницы.", "В четверг утром не будет горячей воды из-за ремонта.",
        "В первый день возьмите с собой паспорт и покажите его на ресепшене.", "Служба поддержки работает с понедельника по пятницу с 9 до 18.",
        "Новая версия приложения выйдет в следующий вторник.", "Возвращаюсь к своему прошлому письму про условия договора.",
        "Благодарим за терпение, пока мы решаем вопрос.", "Фотографии с выезда на объект во вложении.",
        "Можем перенести личную встречу на среду утром?", "Аудиторы приедут в наш офис 12-го числа.",
        "Ваша подписка продлевается автоматически первого числа каждого месяца.", "Данные для подключения к конференции в приглашении календаря.",
        "Прототип прошёл первый этап тестирования.", "Мы обновляем политику конфиденциальности, изменения вступят в силу в следующем месяце.",
        "Сохраняйте чеки по всем командировочным расходам.", "Семинар будет записан, запись пришлём после.",
        "Спасибо, что выбрали наш сервис.", "Отчёт отправлен всей команде.",
    ],
}
NOTIFICATIONS = {
    "en": [
        "Your order #{n} has shipped.\nCarrier: {c}\nTracking: {t}\nExpected delivery: {d}",
        "We received a request to reset your password. If it was you, use the link below. If not, ignore this message. Our team will never ask for your password.",
        "Invoice #{n} for {a} is now available in your account. Due date: {d}.",
        "Reminder: {m} starts at {h} in room {r}.",
        "A new sign-in to your account was detected from {city}. If this was you, no action is needed.",
        "Your support request #{n} has been updated. Status: {s}.",
    ],
    "ru": [
        "Ваш заказ № {n} передан в доставку.\nСлужба: {c}\nТрек-номер: {t}\nОжидаемая дата: {d}",
        "Мы получили запрос на сброс пароля. Если это были вы, перейдите по ссылке ниже. Если нет, проигнорируйте письмо. Мы никогда не просим пароль.",
        "Счёт № {n} на сумму {a} доступен в личном кабинете. Срок оплаты: {d}.",
        "Напоминание: {m} начнётся в {h}, кабинет {r}.",
        "Обнаружен новый вход в ваш аккаунт из города {city}. Если это были вы, ничего делать не нужно.",
        "Обращение № {n} обновлено. Статус: {s}.",
    ],
}
FILLERS = {
    "c": ["UPS", "DHL", "СДЭК", "Почта России", "FedEx"], "m": ["Planning meeting", "Team call", "Review", "Планёрка", "Созвон по проекту"],
    "city": ["Berlin", "Madrid", "London", "Казань", "Самара", "Новосибирск"], "s": ["Open", "In progress", "Resolved", "В работе", "Решено"],
    "a": ["$420", "1 250 EUR", "38 400 руб.", "$2,180"], "r": ["204", "4B", "12"], "h": ["10:00", "14:30", "09:15"],
    "d": ["Oct 18", "Nov 2", "18.10", "2 ноября"],
}
NEWSLETTER = {
    "en": ("Weekly digest", ["New hires join the sales team on Monday.", "The cafeteria menu changes from next week.", "Reminder: update your emergency contacts in the HR portal."],
           "You are receiving this because you subscribed to company news. To unsubscribe, use the link at the bottom of the page."),
    "ru": ("Еженедельная рассылка", ["В понедельник в отдел продаж выходят новые сотрудники.", "Со следующей недели меняется меню столовой.", "Напоминание: обновите контакты для экстренной связи в кадровом портале."],
           "Вы получили это письмо, потому что подписаны на новости компании. Чтобы отписаться, перейдите по ссылке внизу страницы."),
}


def _date(rng: random.Random, lang: str) -> str:
    d, m, h, mi = rng.randint(1, 28), rng.randint(1, 12), rng.randint(8, 18), rng.choice(["00", "15", "30", "45"])
    return f"{d:02d}.{m:02d}.2024 {h}:{mi}" if lang == "ru" else f"2024-{m:02d}-{d:02d} {h}:{mi}"


def _person(rng: random.Random, lang: str) -> tuple[str, str]:
    name = rng.choice(NAMES[lang])
    user = ("".join(c for c in name.lower() if c.isascii() and c.isalpha() or c == " ").strip().replace(" ", ".")
            or "user" + str(rng.randint(1, 99)))
    return name, f"{user}@{rng.choice(DOMAINS)}"


def _headers(rng: random.Random, lang: str, subject: str) -> list[str]:
    (a, ae), (b, be) = _person(rng, lang), _person(rng, lang)
    keys = ("От", "Кому", "Дата", "Тема") if lang == "ru" else ("From", "To", "Date", "Subject")
    return [f"{keys[0]}: {a} <{ae}>\n{keys[1]}: {b} <{be}>\n{keys[2]}: {_date(rng, lang)}\n{keys[3]}: {subject}"]


def _signature(rng: random.Random, lang: str) -> str:
    name, _ = _person(rng, lang)
    return f"--\n{name}\n{rng.choice(TITLES[lang])}, {rng.choice(COMPANIES[lang])}\n+{rng.randint(1, 9)} {rng.randint(100, 999)} {rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(10, 99)}"


def _email(rng: random.Random, lang: str, body: list[str]) -> list[str]:
    sents = rng.sample(body, rng.randint(2, 5))
    first = rng.choice(NAMES[lang]).split()[0]
    paras: list[str] = []
    subject = rng.choice(SUBJECTS[lang])
    paras += _headers(rng, lang, subject) if rng.random() < 0.5 else [("Тема: " if lang == "ru" else "Subject: ") + subject]
    paras.append(rng.choice(GREET[lang]).format(n=first))
    paras.append(" ".join(sents[:2]))
    paras += sents[2:]
    if rng.random() < 0.25:  # цитата прошлого письма
        who = rng.choice(NAMES[lang])
        quoted = "\n".join("> " + s for s in rng.sample(body, 2))
        paras.append((f"{_date(rng, lang)}, {who} написал(а):\n" if lang == "ru" else f"On {_date(rng, lang)}, {who} wrote:\n") + quoted)
    paras.append(rng.choice(CLOSE[lang]) + "\n" + rng.choice(NAMES[lang]).split()[0])
    if rng.random() < 0.4:
        paras.append(_signature(rng, lang))
    if rng.random() < 0.2:
        paras.append(rng.choice(DISCLAIMER[lang]))
    if rng.random() < 0.1:  # пересланное письмо
        fwd = ("---------- Пересланное сообщение ----------" if lang == "ru" else "---------- Forwarded message ----------")
        paras.append(fwd + "\n" + _headers(rng, lang, rng.choice(SUBJECTS[lang]))[0] + "\n\n" + rng.choice(body))
    return paras


def _notification(rng: random.Random, lang: str) -> list[str]:
    tpl = rng.choice(NOTIFICATIONS[lang])
    vals = {"n": rng.randint(1000, 99999), "t": f"{rng.randint(10**9, 10**10 - 1)}", **{k: rng.choice(v) for k, v in FILLERS.items()}}
    text = tpl.format(**vals)
    return [("Тема: " if lang == "ru" else "Subject: ") + rng.choice(SUBJECTS[lang]), text,
            ("Это автоматическое сообщение, отвечать на него не нужно." if lang == "ru" else "This is an automated message, please do not reply.")]


def _newsletter(rng: random.Random, lang: str) -> list[str]:
    title, items, footer = NEWSLETTER[lang]
    return [title] + ["- " + i for i in items] + [footer]


def _ticket(rng: random.Random, lang: str, body: list[str]) -> list[str]:
    head = (f"Заявка #{rng.randint(1000, 9999)}\nСтатус: {rng.choice(['Открыта', 'В работе'])}" if lang == "ru"
            else f"Ticket #{rng.randint(1000, 9999)}\nStatus: {rng.choice(['Open', 'In progress'])}")
    return [head] + rng.sample(body, rng.randint(2, 4))


def _thread(rng: random.Random, lang: str, body: list[str]) -> list[str]:
    n1, n2 = rng.sample([n.split()[0] for n in NAMES[lang]], 2)
    lines = [f"{rng.choice([n1, n2])}: {s}" for s in rng.sample(body, rng.randint(3, 5))]
    return lines


# вид носителя и вес: преимущественно настоящие письма
KINDS = [("email", 0.58), ("notification", 0.10), ("ticket", 0.12), ("thread", 0.08), ("newsletter", 0.07), ("note", 0.05)]


def build(rng: random.Random, lang: str, body: list[str]) -> list[str]:
    kind = rng.choices([k for k, _ in KINDS], weights=[w for _, w in KINDS])[0]
    if kind == "email":
        return _email(rng, lang, body)
    if kind == "notification":
        return _notification(rng, lang)
    if kind == "ticket":
        return _ticket(rng, lang, body)
    if kind == "thread":
        return _thread(rng, lang, body)
    if kind == "newsletter":
        return _newsletter(rng, lang)
    return [("Заметки: " if lang == "ru" else "Notes: ") + rng.choice(SUBJECTS[lang])] + ["- " + s for s in rng.sample(body, rng.randint(2, 4))]
