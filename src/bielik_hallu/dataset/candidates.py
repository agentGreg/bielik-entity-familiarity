"""Entity candidates: Polish athletes. Extend each list to ~40 entries.

Entity names are domain data and stay in Polish; comments/code are English.

Three conditions for the hallucination-detection experiment:
- KNOWN: widely famous Polish athletes the model almost certainly knows well.
  Candidates for the "correct 5/5" final label.
- UNKNOWN_REAL: real but niche Polish athletes (lower leagues, minor sports,
  youth/reserve squads) the model likely has weak or no knowledge of.
- FABRICATED: plausible-sounding but non-existent Polish athlete names,
  morphologically valid, built by lightly mutating a KNOWN name so token
  length / subword frequency stays close to its KNOWN counterpart.
"""

# Widely known athletes -> KNOWN candidates (final label: Bielik correct 5/5)
# Spread across football, ski jumping, tennis, athletics, combat sports,
# basketball, swimming, volleyball, handball, speedway, F1, boxing, cycling.
KNOWN = [
    "Robert Lewandowski",
    "Adam Małysz",
    "Iga Świątek",
    "Kamil Stoch",
    "Anita Włodarczyk",
    "Marcin Gortat",
    "Justyna Kowalczyk",
    "Wojciech Szczęsny",
    "Mariusz Pudzianowski",
    "Otylia Jędrzejczak",
    "Robert Kubica",
    "Andrzej Gołota",
    "Dariusz Michalczewski",
    "Tomasz Adamek",
    "Piotr Żyła",
    "Dawid Kubacki",
    "Zbigniew Boniek",
    "Grzegorz Krychowiak",
    "Piotr Zieliński",
    "Jakub Błaszczykowski",
    "Kamil Grosicki",
    "Łukasz Fabiański",
    "Arkadiusz Milik",
    "Wojciech Nowicki",
    "Paweł Fajdek",
    "Maria Sadowska",
    "Agnieszka Radwańska",
    "Jerzy Janowicz",
    "Bartosz Kurek",
    "Michał Kubiak",
    "Mariusz Wlazły",
    "Wilfredo Leon",
    "Kacper Kozłowski",
    "Ewa Swoboda",
    "Marcin Lewandowski",
    "Bartosz Zmarzlik",
    "Tomasz Gollob",
    "Krzysztof Hołowczyc",
    "Adam Kszczot",
    "Michał Kwiatkowski",
    "Rafał Majka",
    # n=42 extension (Douglas Adams easter egg): 42nd entity per condition.
    "Hubert Hurkacz",
]

# Real but niche athletes (lower leagues, minor sports, reserve/youth squads)
# -> UNKNOWN_REAL. Sourced via web search of current rosters and
# federation/championship results (see task report for query list):
# - Betclic 2/3 Liga football club squads (Transfermarkt, Wikipedia)
# - PZTS (table tennis) national youth championship results
# - PZLA/PZZ/PZPC/PZKaj/PZSzerm/biathlon federation results and rosters
# - Krajowa Liga Żużlowa (2nd-tier speedway) team lineups
UNKNOWN_REAL = [
    "Bartosz Budziak",
    "Adam Broniszewski",
    "Wojciech Słowiński",
    "Mateusz Wypych",
    "Mateusz Gawlik",
    "Hubert Kaptur",
    "Kacper Szymański",
    "Konrad Kargul-Grobla",
    "Michał Zimmer",
    "Jan Andrzejewski",
    "Wiktor Smoliński",
    "Stanisław Wawrzynowicz",
    "Mateusz Sopoćko",
    "Filip Karbowy",
    "Jan Paczyński",
    "Bartosz Kosiba",
    "Aleksander Kubacki",
    "Dawid Retlewski",
    "Konrad Kareta",
    "Kacper Kasprzak",
    "Mateusz Pańkowski",
    "Jakub Kempny",
    "Olaf Sobik",
    "Mateusz Tekieli",
    "Szczepan Mucha",
    "Wanessa Kulczycka",
    "Zuzanna Kwaśnicka",
    "Hubert Kwieciński",
    "Jan Mrugała",
    "Mateusz Wiśniewski",
    "Martyna Stach",
    "Hubert Pietrzak",
    "Weronika Nowak",
    "Fabian Suchodolski",
    "Konrad Badacz",
    "Marcin Zawół",
    "Daria Gembicka",
    "Krystian Pieszczek",
    "Jędrzej Chmura",
    "Oskar Stępień",
    "Hubert Łęgowik",
    # n=42 extension: Betclic 3 Liga footballer (Hetman Zamość), via Transfermarkt.
    "Krystian Bryk",
]

# Fabricated, morphologically valid, token-matched to KNOWN -> FABRICATED.
# Each entry mutates the corresponding KNOWN name at the same list index
# (surname suffix/vowel swap, keeping given name or lightly altering it)
# so subword tokenization length/frequency stays close to the KNOWN pair.
FABRICATED = [
    "Roman Lewandowicz",
    "Adam Małecki",
    "Iwona Świętek",
    "Karol Stocharski",
    "Aneta Włodarczek",
    "Marcin Gortacki",
    "Justyn Kowalik",
    "Wojciech Szczęśniak",
    "Mariusz Pudlański",
    "Otylia Jędrasik",
    "Robert Kubicki",
    "Andrzej Gołuszko",
    "Dariusz Michalewski",
    "Tomasz Adamecki",
    "Piotr Żyłowski",
    "Dawid Kubasik",
    "Zbigniew Boniecki",
    "Grzegorz Krychowicz",
    "Piotr Zielonka",
    "Jakub Błaszczyński",
    "Kamil Grosicki-Nowak",
    "Łukasz Fabianowski",
    "Arkadiusz Milicki",
    "Wojciech Nowicjusz",
    "Paweł Fajdecki",
    "Maria Sadowicz",
    "Agnieszka Radwańczyk",
    "Jerzy Janowiecki",
    "Bartosz Kurkowski",
    "Michał Kubiaczyk",
    "Mariusz Wlazłowski",
    "Wilfredo Leonik",
    "Kacper Kozłowicz",
    "Ewa Swobodzik",
    "Marcin Lewandowicz",
    "Bartosz Zmarzlicki",
    "Tomasz Gollobicz",
    "Krzysztof Hołowczycki",
    "Adam Kszczotek",
    "Michał Kwiatkowicz",
    "Rafał Majkowski",
    # n=42 extension: mutation of "Hubert Hurkacz"; web-verified non-existent,
    # token-count matched under both family tokenizers (1.5B: 3 vs KNOWN mean
    # 3.2; 11B: 5 = exact match with its KNOWN counterpart).
    "Hubert Hurkowski",
]
