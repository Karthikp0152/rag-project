TEST_CASES = [
    {
        "question": "Did Penn cheat on Daria?",
        "type": "multi",
        "reference": (
            "Not exactly - at this point Penn and Daria aren't officially a couple. "
            "Daria confronts Penn about sleeping with Blythe Ortiz and kissing her "
            "while he apparently has a separate girlfriend, but Penn brushes off the "
            "accusation rather than confirming or denying it."
        ),
    },
    {
        "question": "Who is Prichard and what did he do?",
        "type": "multi",
        "reference": (
            "Prichard is the school principal who had an abusive disciplinary "
            "relationship with Daria. The abuse is eventually discovered, and Penn "
            "and others confront him about it."
        ),
    },
    {
        "question": "What did Penn give Daria the first time they met, and what did he refuse from her?",
        "type": "lookup",
        "reference": (
            "Penn gave Daria a piece of orange sea glass, telling her it is the rarest "
            "thing in the world and asking her to owe him something in return. He "
            "refused the ten-dollar bill and the drink she offered him."
        ),
    },
    {
        "question": "Who is Via to Penn, and what happened to her?",
        "type": "multi",
        "reference": (
            "Via is Penn's twin sister. She disappeared from their neighbourhood after "
            "not getting into a ballet program; Penn's stepfather Rhett covered it up "
            "with a story about her moving in with their father and faked paperwork. "
            "Via later lives with the Followhills."
        ),
    },
    {
        "question": "Which college was Penn's dream school, and how did Daria end up there?",
        "type": "multi",
        "reference": (
            "Notre Dame was Penn's dream school, which he pursued on a football "
            "scholarship to escape poverty. Daria followed him to Notre Dame in South "
            "Bend, Indiana, where she joined the cheer team."
        ),
    },
    {
        "question": "Where did Penn live after his mother died?",
        "type": "lookup",
        "reference": (
            "Penn moved in with the Followhills, Daria's family, living under the same "
            "roof as Daria."
        ),
    },
    {
        "question": "What happens between the main characters, Daria and Penn?",
        "type": "synthesis",
        "reference": (
            "Daria and Penn grew up connected through a childhood incident involving "
            "Penn's twin sister Via. Years later, after Penn's mother dies, Daria's "
            "family takes him in, and he becomes her housemate. Their relationship "
            "evolves from rivalry and hostility into romance, complicated by past "
            "trauma, Via's return, and Prichard's abuse of Daria. They eventually "
            "separate for college, then reconcile at Notre Dame."
        ),
    },
    {
        "question": "What position does Penn play in football, and for which school?",
        "type": "lookup",
        "reference": "Penn Scully is the star wide receiver for the Las Juntas football team.",
    },
    {
        "question": "What is Daria's role at her school, and which school does she attend?",
        "type": "lookup",
        "reference": "Daria is the cheer captain at All Saints High.",
    },
    {
        "question": "What is distinctive about the shirts Penn wears?",
        "type": "lookup",
        "reference": "Penn's shirts have a hole right where his heart is.",
    },
    {
        "question": "What car does Daria drive?",
        "type": "lookup",
        "reference": "Daria drives a cherry-red convertible BMW.",
    },
    {
        "question": "What ice cream did Penn order at Gelato Heaven, and why?",
        "type": "lookup",
        "reference": (
            "Penn asked for the most disgusting flavor they had and ordered a scoop "
            "of Key lime pie in a cone, which people say is so sour it makes them sick."
        ),
    },
    {
        "question": "Who are the members of Daria's family?",
        "type": "multi",
        "reference": (
            "Daria's parents are Jaime and Melody (Mel) Followhill, and her sister is Bailey."
        ),
    },
    {
        "question": "Who is Knight?",
        "type": "multi",
        "reference": (
            "Knight Cole is Daria's childhood friend. He lives on the same cul-de-sac, "
            "their parents are best friends, his father is Dean Cole, and he is a "
            "popular jock who plays quarterback."
        ),
    },
    {
        "question": "Why did Daria encourage rumours about her and Principal Prichard?",
        "type": "multi",
        "reference": (
            "After being bullied as a freshman over her parents' reputation, Daria "
            "wanted armor: if people thought the hard-nosed principal had her back, "
            "they would fear her instead of taunting her. So she nurtured the rumors "
            "and kept going to Principal Prichard's office without ever saying they "
            "were dating."
        ),
    },
    {
        "question": "What does Gus claim about Penn and Adriana?",
        "type": "lookup",
        "reference": (
            "Gus claims Penn and Adriana are the golden couple of Las Juntas, have "
            "been seriously together for about two years, and have a baby together."
        ),
    },
    {
        "question": "What did Via say she had been allowed to read for the past four years?",
        "type": "lookup",
        "reference": "Via said she wasn't allowed to read anything but the Bible for the past four years.",
    },
    {
        "question": "What did Penn give Daria on her birthday?",
        "type": "lookup",
        "reference": (
            "At her birthday party, Penn put an orange sea glass necklace on Daria."
        ),
    },
    {
        "question": "What did Gus want after Via gave him Daria's journal?",
        "type": "multi",
        "reference": (
            "Gus wanted Penn to throw the game. Via gave Gus Daria's journal, and Gus "
            "used it as leverage to try to ruin Penn."
        ),
    },
    {
        "question": "What did Penn do to Via when he confronted her about hurting Daria?",
        "type": "lookup",
        "reference": (
            "Penn dragged Via out of the Followhills' house into the backyard and "
            "tossed her into the deep end of the pool, telling her she broke their "
            "deal and blood oath not to touch Daria."
        ),
    },
    {
        "question": "What happened during the football game where Penn was supposed to throw the game?",
        "type": "multi",
        "reference": (
            "Penn deliberately played badly, dropping passes and missing balls so Las "
            "Juntas would lose and Daria would be protected. Knight, the All Saints "
            "quarterback, refused to win as well. Daria ran down to the sidelines and "
            "shouted that if Penn was half the man she knew he was, he would show up "
            "on the field."
        ),
    },
    {
        "question": "How did Daria let Penn know she wanted to talk at his graduation?",
        "type": "lookup",
        "reference": (
            "Daria surprised him at his graduation ceremony, standing in the last row "
            "holding a cardboard sign that said 'Talk?'."
        ),
    },
    {
        "question": "What score did Penn get on his SAT?",
        "type": "unanswerable",
        "expect_abstention": True,
        "reference": "The book does not mention Penn's SAT score.",
    },
    {
        "question": "What instrument does Daria play in the school orchestra?",
        "type": "unanswerable",
        "expect_abstention": True,
        "reference": "The book does not mention Daria playing in a school orchestra.",
    },
    {
        "question": "What is Daria's favourite lasagna recipe?",
        "type": "unanswerable",
        "expect_abstention": True,
        "reference": "The book does not discuss Daria's lasagna recipe.",
    },
]
