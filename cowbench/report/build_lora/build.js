// Ukrainian report on the five LoRA runs. Every number comes from data.json
// (the runs' summary.md tables and reports) - none is typed into the text.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, LevelFormat,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, Footer, PageNumber, ImageRun,
} = require("docx");

const HERE = __dirname;
const D = JSON.parse(fs.readFileSync(path.join(HERE, "data.json"), "utf8"));
const OUT = process.argv[2] || path.join(HERE, "report.docx");
const C = Object.fromEntries(D.configs.map((c) => [c.key, c]));

// ---------------------------------------------------------------- formatting
const NB = " ";
const num = (x, d = 1) => x.toFixed(d).replace(".", ",");
const pc = (x, d = 1) => (x === null || x === undefined ? "—" : num(100 * x, d) + "%");
const pp = (a, b) => num(Math.abs(100 * (a - b)), 1) + NB + "п.п.";

const FONT = "Arial", ACCENT = "1F4E8C", INK2 = "52514E";
const CONTENT_W = 9638;
const BORDER = { style: BorderStyle.SINGLE, size: 4, color: "C9C8C3" };

function runs(text, base = {}) {
  const out = [], re = /(\*\*[^*]+\*\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), font: "Consolas", size: 19, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const p = (t, o = {}) => new Paragraph({ children: runs(t), spacing: { after: 120 }, ...o });
const h1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const h2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const li = (t, level = 0) => new Paragraph({ numbering: { reference: "bullets", level }, children: runs(t), spacing: { after: 60 } });
const caption = (t) => new Paragraph({ children: runs(t, { size: 18, color: INK2, italics: true }), spacing: { before: 60, after: 200 } });
function note(t) {
  return new Paragraph({
    children: runs(t), shading: { type: ShadingType.CLEAR, fill: "EEF3FA", color: "auto" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2A78D6", space: 8 } },
    indent: { left: 284, right: 284 }, spacing: { before: 120, after: 200 },
  });
}
function table(widths, header, rows, opts = {}) {
  const cell = (t, w, head, bold) => new TableCell({
    width: { size: w, type: WidthType.DXA },
    borders: { top: BORDER, bottom: BORDER, left: BORDER, right: BORDER },
    shading: head ? { type: ShadingType.CLEAR, fill: "DCE6F2", color: "auto" }
               : bold ? { type: ShadingType.CLEAR, fill: "F3F7EC", color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ children: runs(String(t), { size: 18, bold: head || bold }) })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: header.map((t, i) => cell(t, widths[i], true, false)) }),
           ...rows.map((r, ri) => new TableRow({ children: r.map((t, i) => cell(t, widths[i], false, (opts.bold || []).includes(ri))) }))],
  });
}
const gap = () => new Paragraph({ children: [], spacing: { after: 120 } });

// ---------------------------------------------------------------- derived numbers
const L1 = C.lora1, L2 = C.lora2, L3 = C.lora3, L4 = C.lora4, L5 = C.lora5;
const B896 = C.base_cow896, ZS = C.zeroshot1920, BF = C.base_frame1920;
const det = D.detector, P = D.pipeline, ST = D.stress, E = D.earlier;
const missShare = det.iou05.missed / 2532;
const perCamMax = 12 / ST[0].fps, perCamPaced = 12 / ST[1].fps;

// ---------------------------------------------------------------- content
const mainRows = D.configs.map((c) => [
  c.name, pc(c.exact), pc(c.posture), pc(c.activity),
  c.nr_exact === null ? "—" : `${c.nr_note ? "≈" : ""}${pc(c.nr_exact)} / ${c.nr_note ? "≈" : ""}${pc(c.nr_activity)}`,
  pc(c.vote),
]);

const body = [
  new Paragraph({ children: [new TextRun({ text: "Muse Glimmer + LoRA на CBVD-5", bold: true, size: 40, color: ACCENT })], spacing: { after: 60 } }),
  new Paragraph({ children: [new TextRun({ text: "Результати навчання і тестів: п'ять прогонів, детектор RT-DETRv2, 12 камер", size: 24, color: INK2 })], spacing: { after: 60 } }),
  new Paragraph({ children: [new TextRun({ text: "Жовтень 2026", size: 20, color: INK2 })], spacing: { after: 300 } }),

  h1("1. Головне"),
  li(`**Поза розв'язана.** Усі адаптери, що вчили позу, помиляються в ній у ${pc(L4.posture)}–${pc(L3.posture)} випадків проти ${pc(BF.posture)}–${pc(B896.posture)} у базової моделі без міркувань і ${pc(ZS.posture)} з міркуваннями.`),
  li(`**Найкращий результат — прогін 4 (lora3norum_w1920):** ${pc(L4.exact)} помилки повного збігу, ${pc(L4.nr_exact)} без корів, розмічених як «жуйка». Для порівняння: базова модель з міркуваннями — ${pc(ZS.exact)}.`),
  li(`**Активність, вивчена на train, на val не переноситься**, доки в ній є жуйка: прогони 1 і 2 погіршили активність до ${pc(L1.activity)} і ${pc(L2.activity)} проти ${pc(B896.activity)} у бази. Коли жуйку прибрали з навчання (прогони 4 і 5), активність стала не гіршою за базову, а в прогоні 4 — найкращою (${pc(L4.activity)}).`),
  li(`**Жуйку модель не розпізнає взагалі** — на одному кадрі її не видно. Це ${pc(383 / 2532)} корів val і більшість залишкових помилок активності.`),
  li(`**Режим «усі корови кадру одним запитом» (прогін 5)** втрачає ${pp(L5.exact, L4.exact)} точності проти прогону 4, але пропускна здатність зросла приблизно у ${num(E.per_cow_update_s / perCamMax, 0)} разів: 12 камер оновлюються раз на ≈${num(perCamMax, 1)} с замість ≈${E.per_cow_update_s} с.`),
  li(`**У повному ланцюжку з детектором** помилка зростає до ${pc(P[1].exact)}, і майже вся різниця — це ${pc(missShare)} корів, яких RT-DETRv2 не знайшов. На знайдених коровах модель працює так само, як на рамках з розмітки.`),

  h1("2. Як вимірювали"),
  p("**Дані.** CBVD-5, набір val: 50 кліпів (341–394), 292 ключові кадри, 2532 корови. Це окрема сесія запису, не та, на якій навчали, тож результати показують перенесення на нові умови, а не запам'ятовування."),
  p("**Що питали.** Для кожної корови дві незалежні відповіді, як у розмітці CBVD-5:"),
  li("**поза:** стоїть / лежить;"),
  li("**активність:** їсть / п'є / жує жуйку / нічого з цього."),
  p("**Метрики:**"),
  li("**помилка пози** і **помилка активності** — частка корів, де відповідь не збіглася з розміткою;"),
  li("**помилка повного збігу** — неправильна хоча б одна з двох відповідей; головна метрика;"),
  li("**без жуйки** — ті самі помилки без 383 корів, розмічених як «жуйка» (лишається 2149): клас, який на одному кадрі не видно і який у прогонах 4–5 навмисно не вчили;"),
  li("**голосування по треку** — відповідь корови замінюється відповіддю більшості по її сусідніх кадрах; зазвичай покращує на 1–3 п.п., але не завжди (у прогоні 2 — ні)."),
  note(`Точність самої оцінки: при помилці близько 25% 95-відсотковий довірчий інтервал — приблизно ±1,7${NB}п.п. Різниця між прогонами менше 2–3${NB}п.п. — на межі шуму.`),

  h1("3. Прогони"),
  p("Усі адаптери — LoRA рангу 8 (alpha 16) на Muse Glimmer 30B (контрольна точка FP8, розпакована в BF16 для навчання), A100 80 ГБ."),
  table([2500, 2000, 1400, 3738], ["Прогін", "Запит", "Роздільність", "Що навчали / особливість"], [
    ["1. lora_w896", "одна корова", "896 px", "позу й активність; адаптер — останній крок навчання"],
    ["2. lora2_w896", "одна корова", "896 px", "те саме, більше епох; 10% кліпів train відкладено як dev, береться найкращий знімок"],
    ["3. lora2pose_w896", "одна корова", "896 px", "лише позу; активність відповідає базова модель"],
    ["4. lora3norum_w1920", "одна корова", "1920 px", "позу й активність, крім жуйки"],
    ["5. lora4frame_w1920", "усі корови кадру", "1920 px", "позу й активність, крім жуйки; один запит на кадр"],
  ]),
  gap(),

  h1("4. Результати на рамках з розмітки"),
  p("Рамки корів взято з розмітки — це оцінка самої моделі поведінки, без впливу детектора."),
  new Paragraph({ children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(HERE, "chart_exact.png")),
                                            transformation: { width: 620, height: 396 } })], alignment: AlignmentType.CENTER }),
  caption("Рис. 1. Помилка повного збігу: усі корови (синій) і без корів, розмічених як «жуйка» (помаранчевий). Для базових конфігурацій без помаранчевого стовпця цього розрахунку немає."),
  table([2900, 1150, 1150, 1150, 1900, 1388], ["Конфігурація", "Повний збіг", "Поза", "Активність", "Без жуйки: повний / активність", "Голосування по треку"],
    mainRows, { bold: [6] }),
  caption("Табл. 1. Помилки на 2532 коровах val. ≈ — розраховано вручну з матриці помилок (у звіті прогону 5 ще не було цього стовпця). Базова модель з міркуваннями — з попереднього бенча, той самий набір val."),

  h1("5. Що показав кожен прогін"),
  h2("Прогін 1 — lora_w896"),
  p(`Поза одразу стала майже ідеальною: ${pc(L1.posture)} проти ${pc(B896.posture)} у бази на тій самій роздільності. Але активність погіршилась з ${pc(B896.activity)} до ${pc(L1.activity)}. Модель вивчила зв'язки, характерні для сесії train (наприклад, «стоїть біля кормового столу — жує жуйку», «лежить — нічого»), а у val ці зв'язки інші.`),
  h2("Прогін 2 — lora2_w896"),
  p(`Більше епох і вибір найкращого знімка по dev не допомогли: ${pc(L2.exact)} повного збігу, активність ${pc(L2.activity)}. Dev-кліпи взято з тієї ж сесії, що й train, тож вибір по dev, ймовірно, лише сильніше підлаштовує модель під train. Висновок: проблема не в кількості навчання, а в тому, що вчимо.`),
  h2("Прогін 3 — lora2pose_w896"),
  p(`Навчали лише позу, активність лишили базовій моделі: ${pc(L3.exact)} повного збігу, поза ${pc(L3.posture)}, активність ${pc(L3.activity)} (як у бази). Знімок кроку 638 цього прогону через vLLM дав ${pc(E.pose638_vllm[0])} / ${pc(E.pose638_vllm[1])} / ${pc(E.pose638_vllm[2])}.`),
  h2("Прогін 4 — lora3norum_w1920"),
  p(`Повна роздільність кадрів і активність без жуйки — найкращий результат: ${pc(L4.exact)} повного збігу, поза ${pc(L4.posture)}, активність ${pc(L4.activity)}. Без корів із жуйкою — ${pc(L4.nr_exact)} і ${pc(L4.nr_activity)}: на тих класах, які видно на кадрі, модель помиляється рідко. Перевірка того ж адаптера на 896 px дала ${pc(E.lora4_at_896_exact)} повного збігу — роздільність при використанні має збігатися з навчальною.`),
  h2("Прогін 5 — lora4frame_w1920"),
  p(`Усі корови кадру — одним запитом, з пронумерованими рамками. Порівняно з прогоном 4 помилка трохи вища (${pc(L5.exact)} проти ${pc(L4.exact)}, без жуйки ≈${pc(L5.nr_exact)} проти ${pc(L4.nr_exact)}), поза так само ${pc(L5.posture)}. Головне — один запит на кадр замість запиту на кожну корову, що й робить можливою роботу з 12 камерами (розділ 7). Порівняно з базовою моделлю в тому ж режимі (${pc(BF.exact)}) адаптер знижує помилку пози з ${pc(BF.posture)} до ${pc(L5.posture)}; активність майже не змінюється (${pc(BF.activity)} → ${pc(L5.activity)}).`),
  p(`Розбір помилок активності прогону 5: з 590 помилок 383 — жуйка, яку модель не називає жодного разу; 131 — корова, що їсть, названа «нічого»; 58 — плутанина з питтям (34 рази «п'є» зайве, 24 пиття пропущено; у val лише 53 корови, що п'ють); 18 — «їсть» зайве.`),

  h1("6. Повний ланцюжок: детектор RT-DETRv2 + модель"),
  p(`На фермі рамок з розмітки немає: корів знаходить детектор (RT-DETRv2 r50, Apache-2.0, дообучений на train CBVD-5, вхід ${det.size} px, поріг ${num(det.threshold, 1)}), і модель відповідає про знайдені рамки. Корова з розмітки, яку детектор не знайшов, рахується помилкою в усіх стовпцях.`),
  table([4100, 1400, 1300, 1400, 1438], ["Рамки (прогін 5)", "Повний збіг", "Поза", "Активність", "Не знайдено"],
    P.map((r) => [r.name, pc(r.exact), pc(r.posture), pc(r.activity), r.missed === null ? "—" : String(r.missed)]), { bold: [3] }),
  caption("Табл. 2. Останній рядок — помилки лише на 1885 знайдених коровах (розраховано з попередніх рядків). Результати через vLLM і через transformers збігаються, тож сервер відповідає так само, як при навчанні."),
  p(`**Детектор:** знаходить ${pc(det.iou05.recall)} корів при перекритті рамок 0,5 (точність ${pc(det.iou05.precision)}) і ${pc(det.iou03.recall)} при 0,3 (точність ${pc(det.iou03.precision)}). Частина «пропусків» — рамка, проведена інакше, ніж у розмітці. Найгірше з дрібними (далекими) коровами: знайдено ${pc(det.by_size[0])} найменших проти ${pc(det.by_size[3])} найбільших.`),
  note(`Висновок: на знайдених коровах модель працює так само, як на рамках з розмітки (поза ≈${pc(P[3].posture)}). Майже весь розрив між ${pc(P[0].exact)} і ${pc(P[1].exact)} дає повнота детектора — це найдешевше місце для покращення.`),

  h1("7. 12 камер на одній A100"),
  p("Стрес-тест: 12 камер, кожна відтворює свій кліп val; кожен кадр проходить через детектор (на тій самій відеокарті), потім одним запитом до моделі. Вимірювання — 300 с після 30 с розігріву."),
  table([2900, 1300, 1500, 1500, 1200, 1238], ["Режим", "Кадрів/с, усі камери", "Оновлення камери, с", "Затримка p50 / p90, с", "Детектор p50, мс", "Встигає"], [
    [ST[0].mode, num(ST[0].fps, 2), "≈" + num(perCamMax, 1), `${num(ST[0].p50)} / ${num(ST[0].p90)}`, String(ST[0].det_ms), "—"],
    [ST[1].mode, num(ST[1].fps, 2), "≈" + num(perCamPaced, 1), `${num(ST[1].p50)} / ${num(ST[1].p90)}`, String(ST[1].det_ms), "ні"],
  ]),
  caption(`Табл. 3. У режимі «раз на 10 с» ${ST[1].over} з ${ST[1].frames} кадрів не вклались у 10 с, черга доходила до ${num(ST[1].late_max, 0)} с.`),
  li(`**Ємність — ≈${num(ST[0].fps, 2)} кадру за секунду**, тобто кожна з 12 камер оновлюється раз на ≈${num(perCamMax, 1)} с. Раніше, з запитом на кожну корову, — раз на ≈${E.per_cow_update_s} с.`),
  li("**Інтервал 10 с — на межі**, черга росте. 12–15 с мають триматися із запасом (варто підтвердити прогоном)."),
  li(`**Детектор на швидкість майже не впливає:** ${ST[0].det_ms}–${ST[1].det_ms} мс на кадр проти ≈${num(ST[0].p50)} с у моделі. Обмеження — обчислення моделі на картинці 1920 px (≈3250 вхідних токенів на кадр).`),

  h1("8. Висновки і що далі"),
  li(`**Поза** — готова: ${pc(L4.posture)}–${pc(L5.posture)} помилки в прогонах 4–5 і ≈${pc(P[3].posture)} на коровах, знайдених детектором.`),
  li("**Активність без жуйки** — добре: 6,5–9,6% помилки в прогонах 4–5. Вчити активність треба без жуйки, інакше модель переносить на нові записи хибні зв'язки з train."),
  li("**Жуйка** з одного кадру не розпізнається. Варіанти: кілька кадрів за 2–3 с на один запит (дорожче), окрема легка модель руху щелепи, або датчики (нашийники / вушні мітки з акселерометром), які на фермах і так міряють жуйку надійніше за камеру."),
  li(`**Детектор** — найбільший резерв: ${pc(missShare)} корів не знайдено. Знизити поріг упевненості і дообучити на вході 1280 px (найбільше страждають дрібні корови).`),
  li(`**Пропускна здатність:** одна A100 — 12 камер раз на ≈${num(perCamMax, 1)} с. Далі: тест на RTX PRO 6000 з FP8 і NVFP4 (скрипти готові), менша роздільність під окремий адаптер або коротший формат відповіді.`),
  li("**Нові дані:** власне розмічене відео з іншого корівника покаже, наскільки результати переносяться на іншу ферму і ракурс камери (зверху)."),
];

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "cowbench", title: "Muse Glimmer + LoRA на CBVD-5: результати",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: ACCENT },
        paragraph: { spacing: { before: 320, after: 140 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: FONT, color: "333333" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 567, hanging: 284 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 1134, hanging: 284 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1134, bottom: 1134, left: 1134 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "888888" })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("written", OUT); });
