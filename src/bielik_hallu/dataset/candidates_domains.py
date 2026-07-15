"""Entity candidates for extension E6: three non-athlete domains
(Polish cities, Polish writers, Polish musicians), 42 entities per condition.

42 entities per condition — chosen deliberately; see Adams (1979).

Entity names are domain data and stay in Polish; comments/code are English.

Mirrors ``candidates.py`` (athletes). Three conditions per domain:
- KNOWN: unquestionably famous Polish entities the models almost certainly
  know well.
- UNKNOWN_REAL: real but niche entities (small villages; obscure published
  writers; obscure real musicians). Every entry was verified to exist via
  live Polish-Wikipedia registry queries (village articles are TERYT-backed;
  people were sampled from stub articles <= 3.5 kB) plus DuckDuckGo spot
  checks. See docs/domain-dataset-validation.md.
- FABRICATED: invented, morphologically valid Polish names built by mutating
  real-name morphology (athletes-list style, e.g. "Roman Lewandowicz").
  Token-matched to the KNOWN list of the same domain under BOTH family
  tokenizers (Bielik-1.5B/4.5B and Bielik-11B/Minitron-7B, which differ;
  see the disambiguation-screening procedure): each 42-name set was selected from a
  larger screened pool by minimising |token-count AUROC(K vs F) - 0.5| under
  both tokenizers. Every fabricated name was screened for non-existence
  against Polish Wikipedia (exact title + search) and web search.

Prompt templates differ per domain type: people domains reuse the athletes
prompt ("Kim jest ..."), the cities domain asks "Czym jest ...". Downstream
extraction must read the template from ``DOMAINS[domain]["prompt_template"]``.
"""

# People domains (writers, musicians) — same template as the athletes run.
PROMPT_TEMPLATE_PEOPLE = "Kim jest {entity}? Odpowiedz jednym zdaniem."
# Place domain (cities/villages/fabricated toponyms).
PROMPT_TEMPLATE_PLACES = "Czym jest {entity}? Odpowiedz jednym zdaniem."

# --------------------------------------------------------------------------
# Domain: cities (polskie miasta / miejscowości)
# --------------------------------------------------------------------------

# Unquestionably famous Polish cities/towns. Composition note: the number of
# names that tokenize to a single token under the 1.5B/4.5B tokenizer is
# capped (9 iconic giants), because fabricated toponyms can never be 1 token;
# the remaining slots hold equally famous multi-token cities/towns so that
# K-vs-F token counts stay matchable (length-matching remedy).
CITIES_KNOWN = [
    "Warszawa", "Kraków", "Wrocław", "Poznań", "Gdańsk", "Szczecin", "Łódź",
    "Katowice", "Gdynia", "Opole", "Zielona Góra", "Gorzów Wielkopolski",
    "Sosnowiec", "Gliwice", "Bytom", "Sopot", "Zakopane", "Płock", "Elbląg",
    "Wałbrzych", "Tarnów", "Koszalin", "Kalisz", "Legnica", "Grudziądz",
    "Słupsk", "Rybnik", "Tychy", "Nowy Sącz", "Bielsko-Biała",
    "Jelenia Góra", "Świnoujście", "Kołobrzeg", "Oświęcim", "Malbork",
    "Zamość", "Przemyśl", "Gniezno", "Wadowice", "Wieliczka", "Augustów",
    "Ciechocinek",
]

# Real but niche villages, harvested from Polish-Wikipedia village categories
# (TERYT-backed; małopolskie / opolskie / podkarpackie voivodeships plus
# powiat słupski — the voivodeship category trees that exist on pl.wiki).
# Existence of every entry verified live against the registry.
CITIES_UNKNOWN_REAL = [
    "Zembrzyce", "Śleszowice", "Kopytówka", "Klecza Górna", "Dobrucowa",
    "Sławkowice", "Nowe Bystre", "Orelec", "Trzemeśnia", "Bysina",
    "Gwoździce", "Wańkowa", "Wasiłowice", "Chrzelice", "Dobcza", "Czerce",
    "Brzeżanka", "Glinik Zaborowski", "Zarębki", "Kłapówka",
    "Ustjanowa Dolna", "Bandrów Narodowy", "Strażów", "Niechobrz",
    "Zagórne", "Łęka Szczucińska", "Lubiczko", "Damienice", "Cikowice",
    "Skomielna Czarna", "Sepnica", "Mniszów", "Lednica Górna", "Jankówka",
    "Częstoszowice", "Wola Dębowiecka", "Sromowce Wyżne", "Łapsze Niżne",
    "Malnia", "Ściborowice", "Łąka Prudnicka", "Hermanowice",
]

# Invented toponyms with valid Polish toponymic morphology (productive
# stems + suffixes: -ice, -ów, -owo, -iny, -niki, -sko, -ęcin, compounds).
# None exists as a locality: screened against Polish Wikipedia (exact title,
# redirect-aware, + intitle search over ~100k TERYT-backed locality articles)
# and web-checked via DuckDuckGo.
CITIES_FABRICATED = [
    "Brzeźów", "Brzozany", "Brzozno", "Cisnik", "Cisogród", "Cisowice",
    "Cyranów", "Czyżewiny", "Dębogród", "Grabnice", "Grabniny", "Grabnowo",
    "Jaworno", "Jaworowiec", "Kaczniki", "Kalinice", "Kawkowice", "Klonice",
    "Kloniec", "Kloniny", "Krzewniny", "Kuniny", "Lipniny", "Lipnowo",
    "Lipnów", "Modrzewów", "Młyniny", "Młynęcin", "Olchęcin", "Pliszków",
    "Sarniec", "Sikorno", "Sikorowice", "Sikorsko", "Szczyglowice",
    "Topolnik", "Wierzbnowa", "Wilgowice", "Wilgowiny", "Wilgów",
    "Wydrzyce", "Wydrów",
]

# --------------------------------------------------------------------------
# Domain: writers (polscy pisarze)
# --------------------------------------------------------------------------

# Canonical, unquestionably famous Polish writers and poets.
WRITERS_KNOWN = [
    "Adam Mickiewicz", "Juliusz Słowacki", "Henryk Sienkiewicz",
    "Bolesław Prus", "Władysław Reymont", "Stefan Żeromski",
    "Maria Konopnicka", "Eliza Orzeszkowa", "Cyprian Kamil Norwid",
    "Jan Kochanowski", "Ignacy Krasicki", "Aleksander Fredro", "Adam Asnyk",
    "Julian Tuwim", "Wisława Szymborska", "Czesław Miłosz",
    "Zbigniew Herbert", "Tadeusz Różewicz", "Witold Gombrowicz",
    "Bruno Schulz", "Stanisław Lem", "Olga Tokarczuk", "Andrzej Sapkowski",
    "Ryszard Kapuściński", "Sławomir Mrożek", "Stanisław Wyspiański",
    "Stanisław Ignacy Witkiewicz", "Kornel Makuszyński", "Jan Brzechwa",
    "Aleksander Kamiński", "Gustaw Herling-Grudziński",
    "Jarosław Iwaszkiewicz", "Konstanty Ildefons Gałczyński",
    "Krzysztof Kamil Baczyński", "Halina Poświatowska", "Agnieszka Osiecka",
    "Dorota Masłowska", "Marek Hłasko", "Tadeusz Konwicki",
    "Zofia Nałkowska", "Maria Dąbrowska", "Adam Zagajewski",
]

# Real but obscure published writers, sampled from Polish-Wikipedia stub
# articles (<= 3.5 kB) in writer categories (folk poets, 20th-century poets
# and prose writers). Existence of every entry verified live.
WRITERS_UNKNOWN_REAL = [
    "Sebastian Lesiczka", "Ignacy Ślusarczyk", "Gotfryd Bendziułła",
    "Tobiasz Stullich", "Jan Luśtych", "Emilia Michalska",
    "Frycz Olszewski", "Janina Boniakowska", "Bronisława Kozłowska",
    "Wincenty Motas", "Jan Marczówka", "Stefania Matysiewicz",
    "Antoni Juroszek", "Augustyn Piecha", "Izydor Wilk",
    "Rozalia Grzegorczyk", "Stanisław Buczyński", "Jan Mehl",
    "Barbara Krajewska", "Roman Drejza", "Franciszek Magryś",
    "Henryk Biłka", "Krzysztof Coriolan", "Wincenty Faber",
    "Zdzisław Ćmoch", "Zygmunt Fijas", "Edward Hołda", "Marta Berowska",
    "Bernard Antochewicz", "Włodzimierz Chełmicki", "Ryszard Binkowski",
    "Irena Dowgielewicz", "Władysław Bochenek", "Juliusz Dankowski",
    "Jerzy Gałuszka", "Andrzej Gerłowski", "Stanisław Benski",
    "Olga Chrobra", "Izabella Bielińska", "Kazimiera Dębska",
    "Marcelina Grabowska", "Sylwester Banaś",
]

# Fabricated, morphologically valid, token-matched to WRITERS_KNOWN.
# Mostly light mutations of famous writer surnames (suffix/vowel swaps),
# athletes-list style, plus invented names for token-length variety.
WRITERS_FABRICATED = [
    "Adam Asnykowski", "Adam Miczewski", "Adam Rzepik", "Adam Turkot",
    "Adrian Sapkowiak", "Agata Osieńska", "Aleksander Fredowski",
    "Andrzej Sapkowicz", "Antoni Fredecki", "Bernard Szulczyński",
    "Bogdan Pruszczyński", "Bolesław Prusek", "Bolesława Trzemeska",
    "Czesław Miłoszak", "Czesław Norwidzki", "Dorota Masłowicz",
    "Eliza Orzeszkiewicz", "Ewa Sarniak", "Grzegorz Herlingiewicz",
    "Henryk Sienkowski", "Ignacy Krasowicz", "Jakub Kochanecki",
    "Jan Micewicz", "Jarosław Iwaszkowski", "Julian Tuwiński",
    "Juliusz Słowecki", "Józef Brzechwiński", "Kamil Baczkowski",
    "Konstanty Gałczewski", "Magdalena Dąbrowicka", "Marek Dulik",
    "Maria Dąbkowiecka", "Marta Wilczak", "Olga Tokarowska",
    "Piotr Sowicz", "Piotr Wronowik", "Prakseda Wojniczak",
    "Stanisław Lemowski", "Szymon Żeromczyk", "Sławomir Mrożewski",
    "Tadeusz Konwiński", "Władysław Remski",
]

# --------------------------------------------------------------------------
# Domain: musicians (polscy muzycy)
# --------------------------------------------------------------------------

# Unquestionably famous Polish musicians: composers, jazz, classical
# performers, singers and band leaders.
MUSICIANS_KNOWN = [
    "Fryderyk Chopin", "Stanisław Moniuszko", "Karol Szymanowski",
    "Witold Lutosławski", "Krzysztof Penderecki", "Henryk Mikołaj Górecki",
    "Wojciech Kilar", "Zbigniew Preisner", "Krzysztof Komeda",
    "Tomasz Stańko", "Michał Urbaniak", "Urszula Dudziak", "Leszek Możdżer",
    "Ignacy Jan Paderewski", "Artur Rubinstein", "Krystian Zimerman",
    "Rafał Blechacz", "Czesław Niemen", "Marek Grechuta",
    "Zbigniew Wodecki", "Krzysztof Krawczyk", "Maryla Rodowicz",
    "Anna German", "Ewa Demarczyk", "Anna Jantar", "Irena Santor",
    "Andrzej Zaucha", "Ryszard Riedel", "Tadeusz Nalepa",
    "Grzegorz Ciechowski", "Kora Jackowska", "Kazik Staszewski",
    "Muniek Staszczyk", "Agnieszka Chylińska", "Edyta Górniak",
    "Justyna Steczkowska", "Beata Kozidrak", "Dawid Podsiadło",
    "Michał Szpak", "Paweł Kukiz", "Edyta Bartosiewicz",
    "Ryszard Rynkowski",
]

# Real but obscure musicians, sampled from Polish-Wikipedia stub articles
# (<= 3.5 kB) in musician categories (folk musicians, violinists, organists,
# classical composers). Existence of every entry verified live.
MUSICIANS_UNKNOWN_REAL = [
    "Mateusz Dziubiński", "Karol Mikuszewski", "Marian Bujak",
    "Antoni Cichecki", "Jarosław Śliwa", "Bronisław Zaremba",
    "Bartosz Niedźwiecki", "Danuta Głowacka", "Kazimierz Olechowski",
    "Józef Bryjka", "Ludwik Holcman", "Małgorzata Górnisiewicz",
    "Katarzyna Bąkowska", "Roland Orlik", "Jacek Niwelt",
    "Bogusław Bruczkowski", "Izabela Ceglińska", "Michał Grabarczyk",
    "Cezary Czternastek", "Natalia Juśkiewicz", "Joanna Giemzowska",
    "Kazimierz Baranowski", "Roksana Kwaśnikowska", "Bazyli Bohdanowicz",
    "Joanna Marciniak", "Ludwik Gawroński", "Magdalena Malec",
    "Jakub Moneta", "Edward Forski", "Leszek Woś", "Jerzy Kukla",
    "Kazimierz Madziała", "Antoni Karnaszewski", "Andrzej Dorawa",
    "Grzegorz Górkiewicz", "Ireneusz Wyrwa", "Józef Chwedczuk",
    "Jan Drzewoski", "Henryk Klaja", "Józef Krzeczek", "Bartosz Izbicki",
    "Adam Prucnal",
]

# Fabricated, morphologically valid, token-matched to MUSICIANS_KNOWN.
# Mostly light mutations of famous musician surnames, athletes-list style.
MUSICIANS_FABRICATED = [
    "Adam Rylec", "Alicja Jantarek", "Anna Germańska",
    "Antoni Zauszyński", "Artur Rubinowski", "Damian Podsiedlik",
    "Edyta Górniewicz", "Ewa Demarska", "Ewa Tarnowiak",
    "Franciszek Chopieński", "Fryderyk Szopenowicz", "Grzegorz Ciechowicz",
    "Henryk Góreczak", "Ignacy Paderowski", "Iwona Santorek", "Jan Basiór",
    "Jan Fidor", "Justyna Steczyńska", "Karol Szymanowicz",
    "Konrad Penderak", "Konstanty Staszczykowski", "Krzysztof Komedowski",
    "Krzysztof Krawiecki", "Leonard Barcik", "Marcin Urbaniecki",
    "Marek Grechowski", "Marek Surmik", "Marlena Rodowska",
    "Marta Klawecka", "Michał Szpakiewicz", "Mikołaj Szpaczek",
    "Mirosław Staszczyński", "Patryk Kukizek", "Paweł Kukizewski",
    "Piotr Dygoń", "Rafał Blechowski", "Remigiusz Blecharczyk",
    "Stanisław Moniuszewski", "Tomasz Stańkowski", "Witold Lutosiński",
    "Wojciech Kilarowski", "Zbigniew Prajsner",
]

# --------------------------------------------------------------------------
# Public structure consumed by downstream extraction.
# --------------------------------------------------------------------------
DOMAINS = {
    "cities": {
        "KNOWN": CITIES_KNOWN,
        "UNKNOWN_REAL": CITIES_UNKNOWN_REAL,
        "FABRICATED": CITIES_FABRICATED,
        "prompt_template": PROMPT_TEMPLATE_PLACES,
    },
    "writers": {
        "KNOWN": WRITERS_KNOWN,
        "UNKNOWN_REAL": WRITERS_UNKNOWN_REAL,
        "FABRICATED": WRITERS_FABRICATED,
        "prompt_template": PROMPT_TEMPLATE_PEOPLE,
    },
    "musicians": {
        "KNOWN": MUSICIANS_KNOWN,
        "UNKNOWN_REAL": MUSICIANS_UNKNOWN_REAL,
        "FABRICATED": MUSICIANS_FABRICATED,
        "prompt_template": PROMPT_TEMPLATE_PEOPLE,
    },
}
