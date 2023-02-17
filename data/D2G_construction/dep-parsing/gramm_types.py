__author__ = 'ASU'

from typing import Optional, Union

GRAMMATICAL_RELATIONS = [
    ('pred', 'predicate'),
    ('aux', 'auxiliary'),
    ('auxpass', 'passive auxiliary'),
    ('cop', 'copula'),
    ('conj', 'conjunct'),
    ('cc', 'coordination'),
    ('punct', 'punctuation'),
    ('arg', 'argument'),
    ('subj', 'subject'),
    ('nsubj', 'nominal subject'),
    ('nsubjpass', 'nominal passive subject'),
    ('csubj', 'clausal subject'),
    ('csubjpass', 'clausal subject'),
    ('comp', 'complement'),
    ('obj', 'object'),
    ('dobj', 'direct object'),
    ('iobj', 'indirect object'),
    ('pobj', 'prepositional object'),
    ('pcomp', 'prepositional complement'),
    ('attr', 'attributive'),
    ('ccomp', 'clausal complement'),
    ('xcomp', 'xclausal complement'),
    ('complm', 'complementizer'),
    ('mark', 'marker'),
    ('rel', 'relative'),
    ('ref', 'referent'),
    ('expl', 'expletive'),
    ('acomp', 'adjectival complement'),
    ('mod', 'modifier'),
    ('advcl', 'adverbial clause modifier'),
    ('purpcl', 'purpose clause modifier'),
    ('tmod', 'temporal modifier'),
    ('rcmod', 'relative clause modifier'),
    ('amod', 'adjectival modifier'),
    ('num', 'numeric modifier'),
    ('number', 'compound number modifier'),
    ('quantmod', 'quantifier modifier'),
    ('nn', 'noun compound modifier'),
    ('appos', 'appositional modifier'),
    ('abbrev', 'abbreviation modifier'),
    ('partmod', 'participial modifier'),
    ('infmod', 'infinitival modifier'),
    ('advmod', 'adverbial modifier'),
    ('neg', 'negation modifier'),
    ('measure', 'measure-phrase'),
    ('det', 'determiner'),
    ('predet', 'predeterminer'),
    ('preconj', 'preconjunct'),
    ('poss', 'possession modifier'),
    ('possessive', 'possessive modifier'),
    ('prep', 'prepositional modifier'),
    ('prt', 'phrasal verb particle'),
    ('parataxis', 'parataxis'),
    ('sdep', 'semantic dependent'),
    ('xsubj', 'controlling subject'),
    ('agent', 'agent'),
    ('conj', 'conj_collapsed'),
    ('prep', 'prep_collapsed'),
    ('prepc', 'prepc_collapsed'),
    
    ('conj_+',
     "[subject,verb] '' One peculiar thing about the grammar on the title is that instead of being just a normal independent clause , which is subject + verb it turns out to be verb + subject ."),
    ('conj_and',
     '[friendship,altered] An exploration of friendship and altered states of normality within a broken world of deals and highs .'),
    ('conj_and\\/or',
     '[show,physically] Why do people want to be on that show to either humiliate their selves , or be hurt emotionally and\\/or physically ?'),
    ('conj_as',
     '[is,is] For positive changes Mrs. Phillips feels that there is better communication like cell phones and email but as for negative changes , she is troubled that people email constantly or text rather that using just a phone or writing a letters .'),
    ('conj_but',
     "[passion,connection] It was not Govinda 's passion to become a Samana or his need to experience something new that made him decide to leave the village , but it was the connection between Siddhartha and Govinda , the shadow-life Govinda has always known , that made the decision for him ."),
    ('conj_et', '[Dulce,Decorum] Dulce et Decorum Est by Wilfird Owen is written about the First World War .'),
    ('conj_if',
     "[litany,resolution] Thus , while the result seems a foregone conclusion , Keats ' static world creates a litany of possible outcomes more beautiful than if any final resolution ."),
    ('conj_just',
     '[birth,men] Also it is pointed out when women were giving birth it was like a battle , just as painful as the ones men fought in wars .'),
    ('conj_loved',
     "[that,Caesar] '' ... not that I loved Caesar less , but that I loved Rome more '' -LRB- III . ii .21 -RRB- ."),
    ('conj_merely',
     "[home,housewives] On the other hand , the feminine roles symbolize the patriarchal nature of the traditional values of society ; in the text the woman 's place was the home , as merely housewives , this is implied in the fact that they were `` wearing faded house dresses and sweaters '' ."),
    ('conj_negcc',
     "[more,God] Antonio believes more of the Golden Carp rather than God because God punished people while the Golden Carp `` swallows everything good and evil `` and `` becomes `` a new sun to shine its good light upon the earth '' -LRB- 176 -RRB- ."),
    ('conj_nor',
     "[incest,bigamy] Homosexuality is n't the same as incestuous behavior ; making same-sex marriage legal does n't condone incest nor bigamy ."),
    ('conj_only', '[innate,come] The idea can not be innate , but only come from experiences .'),
    ('conj_or',
     "[work,even] This is incorrectly assumed , people do n't know whether those couples would be more likely to work or even place a higher standard on marriage ."),
    ('conj_so',
     "[leading,woman] The novel 's influence was so pronounced around the country , leading to and during the civil war , that upon meeting Harriet Beech Stowe , President Abraham Lincoln said , `` So you are the woman who started this war ."),
    ('conj_times',
     '[conversation,Margrethe] The author Michael Frayn also appeared to not include exits or entrances in the play , as Copenhagen was a single conversation between the two Bohr and Heisenberg and at times Margrethe .'),
    ('conj_v.',
     "[Brown,Board] Although Jim Crow laws were settled by the 1954 court case , Brown v. Board of Education of Topeka , where all laws and public policy based on the theory of `` separate but equal '' were deemed unconstitutional ; they were not fully eliminated until the mid 1960 's , almost one hundred years after the end of U.S. Civil War and the beginning of Radical Reconstruction ."),
    ('conj_versus', "[Tom,Bob] This trial is Tom versus Bob Ewell on charges that Tom rapped Bob 's daughter ."),
    ('conj_vs.', "[Chiang,Mao] '' -LRB- Civil War : Chiang vs. Mao -RRB- ."),
    ('conj_whether',
     '[disputed,gets] There were many small arguments during 1885 to 1914 , as countries disputed over whether who gets which country .'),
    ('conj_yet',
     "[delicious,unhealthy] People in other countries , many times have to conserve their food and get whatever they can when it is available shows us that many times they do n't get the delicious , yet very unhealthy , food that we have ."),
    ('dep', "[An,'s] Analysis of Nicola Monaghan 's The Killing Jar An engaging read ."),
    ('prep_#',
     "[is,One] '' # `` One other thing schools are trying to do is : taking the auditorium and making them into large classrooms for one class that contains the most children ."),
    ('prep_694',
     "[York,creation] -LRB- New York : Bedford \\/ St. Martin 's 2010 -RRB- 694 The creation of a minimum wage law now prevented companies from paying unfair wages to its employees ."),
    ('prep_aboard',
     '[236,Pequod] 236 -RRB- After capturing a whale , the crew aboard the Pequod throws the remains of the whale out for the sharks to eat .'),
    ('prep_about',
     '[say,experiences] But I suspect that this may say more about my experiences in life and does not detract from the content that is disturbing in parts .'),
    ('prep_above',
     '[mentioned,leads] As I mentioned above the leads , wires and electrodes must be MR compatible to avoid radiofrequency -LRB- RF -RRB- heating and burning .'),
    ('prep_according_to',
     '[need,Epicurus] According to Epicurus to truly be happy we need friends , freedom from social pressures , and the ability to contemplate our main sources of anxieties .'),
    ('prep_across',
     '[schools,States] For the public schools of Louisiana and across the United States I would propose the expansion of healthcare services provided by the school , an increase in after school and community programs , and the use of a comprehensive curriculum .'),
    ('prep_across_from',
     '[sitting,you] The theater was very small and there were seats on all sides of the stage which was very interesting and you could see the audience reaction sitting across from you .'),
    ('prep_after',
     '[taking,mountain] I love curling up by the fireplace , or taking a hot bath after the mountain of snow you just had to shovel through .'),
    ('prep_against',
     '[is,beliefs] Many oppose the idea of allowing homosexual marriages , in truth it is against religious beliefs .'),
    ('prep_ahead_of',
     '[been,time] The novel To Kill A Mockingbird by Harper Lee is thought to have been way ahead of her time of literature and went places were books of that time did not go , and still do not go .'),
    ('prep_along',
     "[picture,band] A persuasive presentational device used in the text is the picture of the pants with `` Make Poverty History '' along the band ."),
    ('prep_along_with',
     '[determine,income] Your credit score along with your available income will determine what kind of interest rate you will have and how much money you can spend per month on a house payment .'),
    ('prep_alongside',
     "[effemanite,comment] '' This effemanite description alongside the comment of `` his freend and compeer , '' suggests to the audience that he is perhaps homosexual ."),
    ('prep_although',
     '[thinking,women] Different with women who has more mature thinking although women still in the young ages .'),
    ('prep_amendment',
     "[first,states] The first amendment states '' . . respecting the establishment of religion ... '' When Christian students listen and are forced to learn the theory of Evolution , it is restricting them to worship without obstacles and is therefore , disrespecting the establishment of religion by defying the existence of God ."),
    ('prep_amid',
     "[were,war] Lincolns ' Gettysburg Address on the other hand was created for the people of the United States while they were amid the Civil war ."),
    ('prep_among',
     '[difficult,heterosexuals] It is difficult among heterosexuals , and automatically thought as less able to be performed by same-sex couples .'),
    ('prep_amongst',
     '[Texting,drivers] Texting while driving has developed into a growing danger amongst many drivers .'),
    ('prep_apart_from',
     '[tell,all] When I got outside Ms. Tanya said hello and insisted that I just call hero she could tell my voice apart from all of the children .'),
    ('prep_around',
     '[everything,you] A time where you wake up snuggled up with the heat blaring , a time where everything around you takes your breath away and a time where what you have is cherished most .'),
    ('prep_as', '[read,comment] So I guess that for me this novel read as a social comment .'),
    ('prep_as_for',
     '[groups,profile] As for the profile of major players in the derivatives markets , individual investors and securities firms are the two dominant groups of investors .'),
    ('prep_as_from', '[save,extension] There most likely will not we another race to save as from extension .'),
    ('prep_as_of',
     '[schooled,Spring] As of Spring 2007 , an estimated 2.9 % of children are being home schooled in the United States alone .'),
    ('prep_as_per',
     '[parent,advice] As per the advice of a friend , when I had my first child , I became a single parent and she encouraged me to applied for temporary government assistance -LRB- TANF -RRB- .'),
    ('prep_as_to',
     '[granted,marriages] This argument hinged upon California Family Code Section 297.5 , which granted the same rights and responsibilities to civil unions and domestic partnerships as to marriages .'),
    ('prep_aside_from',
     '[influenced,orders] Aside from the orders of her Psychiatrist , her impulsiveness due to her mental instability and pressure from friends such as Sylvia Plath may have influenced her to publicize her innermost sentiments in order to embrace them more fully .'),
    ('prep_at',
     '[is,question] Their ability to thrive in society is also at question , they are being ostracized as incapable of emotional attachment and interaction due to sexuality .'),
    ('prep_away_from',
     "[stayed,topic] Lee went head at the lifestyle of the 1930 's in Alabama , and talked about racism and prejudice as many others stayed away from that topic ."),
    ('prep_barring',
     '[grades,segregation] Under the 10th Amendment of the US Constitution , education evolved into an escalade of state , county and city districts as funds for all school grades and laws barring segregation benefited scores of students .'),
    ('prep_based_on',
     '[inheritance,blood] To ensure this , monogamy replaced free sexual relations , and inheritance based on blood has been formed .'),
    ('prep_because',
     "[believes,God] Antonio believes more of the Golden Carp rather than God because God punished people while the Golden Carp `` swallows everything good and evil `` and `` becomes `` a new sun to shine its good light upon the earth '' -LRB- 176 -RRB- ."),
    ('prep_because_of',
     "[subject,orientation] Church and state should be kept separate ; all people are created equally and should n't be held to any different standards on any subject because of their own personal orientation ."),
    ('prep_before',
     '[night,arrival] In the first dream , the night before the arrival of Ultima , Antonio is born and both sides of his family gather together for the arrival of the baby boy .'),
    ('prep_beginning',
     "[Antonio,passage] `` O '' -LRB- 2.1.252 -RRB- laments Antonio beginning the passage with an informal introduction ."),
    ('prep_behind', '[faded,Siddhartha] The shadow soon faded behind Siddhartha when Govinda became a man .'),
    ('prep_below',
     "[50,timers] London tells of the traveler 's lack of concern for important survival instructions when he states , `` The old timer had been very serious in laying down the law that no man must travel alone in the Klondike after 50 below '' those old timers were rather womanish , some of them , he thought '' -LRB- 12 -RRB- ."),
    ('prep_beneath',
     '[unmoving,red] Within a second , gunshots echo ; and a fallen beauty lays unmoving , painting the snow beneath a deep red .'),
    ('prep_beside', '[was,myself] I was beside myself .'),
    ('prep_besides', '[think,fact] Besides the fact that they act similarly , they also think similarly .'),
    ('prep_between', '[ritual,man] It is widely recognized as a ritual between both man and woman .'),
    ('prep_beyond',
     "[look,horizon] Yet , its concept is still recycled in the twenty-first century , as it inspires all humanity to look beyond the `` horizon , '' as Janie explains ."),
    ('prep_boys',
     '[play,13] Among elementary and middle school populations , girls play video games for an average of about 5.5 h\\/week and boys average 13 h\\/week -LRB- Anderson et al.'),
    ('prep_but',
     "[separate,equal] Although Jim Crow laws were settled by the 1954 court case , Brown v. Board of Education of Topeka , where all laws and public policy based on the theory of `` separate but equal '' were deemed unconstitutional ; they were not fully eliminated until the mid 1960 's , almost one hundred years after the end of U.S. Civil War and the beginning of Radical Reconstruction ."),
    ('prep_by', '[feel,content] Some people may feel shocked by the content of this novel .'),
    ('prep_by_means_of',
     '[shows,exemplification] What these examples of the optimal gauge of the absurd teaches us -LRB- or merely shows us indirectly , by means of exemplification -RRB- is that no piece of absurd literature should be shorter than a few dozen pages .'),
    ('prep_close_to',
     "[draw,message] The ability to listen with one 's eyes , nose , and fingertips is an indispensable skill in the practice of medicine because it allows us to quickly draw close to the message being communicated ."),
    ('prep_compared_to', "[have,women] No man would want to have his chest compared to a women 's ."),
    ('prep_compared_with',
     "[are,truths] Similarly in `` The World According to Monsanto , '' the irony of the fact that new dyes added to foods must be tested for months whereas new genes that could potentially kill humans can bypass all extensive testing shocks the reader , and evokes a similar response , in which previous misconceptions are compared with truths ."),
    ('prep_concerning',
     '[issue,everybody] Although this ad uses the example of a middle aged , new father , obesity is an issue concerning everybody .'),
    ('prep_considering',
     "[asked,nobody] '' Considering nobody has asked him a question this makes it blatantly obvious from the beginning of the story that the narrator is in fact mentally disturbed ."),
    ('prep_contrary_to',
     "[Magazine,nightmare] -LRB- Magazine -RRB- Contrary to the horrid and stupidly abstract nightmare that these parents envision student interaction to be , the majority of public school students come out of the schools ' social environment virtually unscathed and both mentally and physically intact ."),
    (
        'prep_depending',
        '[change,circumstances] These feelings can change quite quickly depending outer circumstances .'),
    ('prep_depending_on',
     '[carve,recourses] However , artists also carve from softer substances such as wood and soap depending on recourses and cost .'),
    ('prep_despite',
     "[chose,decision] '' When Siddhartha announced he was leaving the Buddha , Govinda tried to convince him not to , but despite Siddhartha 's decision , Govinda chose his own path ."),
    ('prep_down',
     '[steers,paths] This takes deep roots in Hamlets persona and steers his life down paths that would otherwise have been avoided .'),
    ('prep_due_to', '[known,this] Due to this , they both have never known a life without each others company .'),
    ('prep_during',
     '[nurse,pregnancy] By using the fact that female are the ones who nurse children because of their enlarged breasts during pregnancy , it is suggesting that the father is obese because his child believes he is the mother and has large enough breasts to feed on .'),
    ('prep_en',
     "[difeme,profondeur] To reduce wastage , the number of soldiers killed during a normal day , the French adopted the use of `` difeme en profondeur -LRB- defense in depth -RRB- '' '' -LRB- Smith 195 -RRB- ."),
    ('prep_except',
     '[hold,that] They were not able to vote , hold bank accounts , sign contracts , or hold a professional position except that of a teacher .'),
    ('prep_except_for',
     "[unfaithful,handkerchief] Iago preys on Othello 's vulnerability leading him to believe that Desdemona has been unfaithful without significant evidence , except for her handkerchief ."),
    ('prep_far_from', '[was,anything] The Samana way of life was far from anything Siddhartha and Govinda knew .'),
    ('prep_followed_by',
     "[adding,servants] Portia finely says that she thinks worthy of Bassanio and Shakespeare has made the audience excited by adding the sound of trumpets and the Prince of Moroccobeing followed by he 's servants that makes the audience think that he 's rich ."),
    ('prep_following',
     "[lines,them] Sexton 's apparent dislike of change is apparent when juxtaposing lines such as , `` The objects keep changing , '' and , `` Nothing is what is seems to be , '' to the lines immediately following them , which respectively state , `` Ashtrays to cry into , '' and , `` My objects dream and wear new costumes ."),
    ('prep_for', '[this,me] And this , for me , made the novel convincing and rather clever .'),
    ('prep_forth',
     "[furnish,mother] Hamlet disapproves of how his mother was so quick to mourn the death of his father that `` The funeral baked meats\\/Did coldly furnish forth -LRB- his -RRB- mother 's wedding '' -LRB- I.ii.183-84 -RRB- ."),
    ('prep_from',
     '[describes,Kerrie-Ann] She describes her spiral from a five year old Kerrie-Ann having intelligent exchanges with a neighbor , into a playground drug dealing Kez and a horrific revenger .'),
    ('prep_genetic',
     '[helps,engineering] Genetic engineering , it helps promote agriculture business , which can cause the use of more Herbicides , which tend to contaminate water and cause farmers illness .'),
    ('prep_half',
     "[\\/,love] Furthermore , in reality , according to Elizabethan beliefs , daughters were supposed to `` carry \\/ Half -LRB- their -RRB- love with -LRB- their husbands -RRB- , half -LRB- their -RRB- care and duty '' -LRB- I.i.103-104 -RRB- , since it was the duty of daughters and sons to love their father as well as their spouses and children ."),
    ('prep_if',
     '[left,England] The first and most famous settlers were the Pilgrims , who left Southampton in England in 1620 for the New World due to religious differences with Church if England , the governing body of Christianity in England .'),
    ('prep_in',
     '[experiences,life] But I suspect that this may say more about my experiences in life and does not detract from the content that is disturbing in parts .'),
    ('prep_in_accordance_with',
     "[laws,income] Health insurance subsidies implicit in the federal tax laws in accordance with the individual 's wage income will need to pay Social Security payroll tax and personal income tax ."),
    ('prep_in_addition_to',
     '[is,species] This is in addition to the 122 species thought to have gone extinct since 1980 -LRB- 80 -RRB- .'),
    ('prep_in_case_of',
     '[children,emergencies] Yes cell phones are great to have for children in case of emergencies , but there are other ways .'),
    ('prep_in_front_of', '[occur,him] In the very last dream , he witnesses three deaths that occur in front of him .'),
    ('prep_in_lieu_of',
     "[drugs,language] If educators took a poll of students ' choices , would they teach sex , drugs and rock n'roll in lieu of language and math ?"),
    ('prep_in_place_of',
     '[are,land] The massive barriers of China forced people to live on the east side of China , because many of the places in China are ethier inhabitable , or that there are ethier platus , moutains , deserts , or basins in place of clear land for people to live on it .'),
    ('prep_in_spite_of',
     '[liked,this] In spite of this , Betty Smith liked poor people because poor people appreciated every little item they may have .'),
    ('prep_including',
     '[everyone,herself] She shows that everyone , including herself , becomes fearful when people of her ethnicity enter a neighborhood that is not Mexican .'),
    ('prep_inside',
     '[has,she] The immigrant from Mexico appears shy and terrified because inside she truly has reasons to feel that way .'),
    ('prep_inside_of',
     '[come,US] People inside of US come up with plenty of ideas to prevent illegal entry ; building up a border fence is one of them .'),
    ('prep_instead_of',
     "[say,hearing] '' I took a couple of tries but eventually instead of hearing him say , `` Wittle Wamb '' I heard , `` Little Lamb ."),
    ('prep_into',
     '[describes,drug] She describes her spiral from a five year old Kerrie-Ann having intelligent exchanges with a neighbor , into a playground drug dealing Kez and a horrific revenger .'),
    ('prep_involving',
     '[scandal,House] Haldeman and John Ehrlichman and Attorney General Richard Kleindienst resign due to the scandal involving the White House .'),
    ('prep_like',
     '[followed,shadow] Like a shadow to Siddhartha , Govinda has always followed and pursued whatever it was that Siddhartha was doing .'),
    ('prep_near',
     '[venture,settlements] This caused wolves to venture near human settlements more often , looking for some kind of nourishment .'),
    (
        'prep_next',
        "[years,November] `` Five years next November '' -LRB- 87 -RRB- he told Daisy when they met again ."),
    ('prep_next_to',
     '[bed,husband] She wakes up one morning and sees the plastic toy then throws it on the bed next to her husband to try and scare him .'),
    ('prep_not', "[are,wow] '' Not `` wow , there are so many ways to waste this empty space !"),
    ('prep_notwithstanding',
     "[fights,obedience] '' Notwithstanding her obedience , Janie fights with her husband internally , as she can never express to him who she really is ."),
    ('prep_of', "[Analysis,An] Analysis of Nicola Monaghan 's The Killing Jar An engaging read ."),
    ('prep_off',
     '[get,bus] Tanya quickly scribbled a name and a bus number on my hand and said , `` Ok , go get him off the bus .'),
    ('prep_off_of',
     '[pick,rug] After the battle has ended , the boys are allowed to pick bills and coins off of a rug .'),
    ('prep_on', '[story,drugs] A love story on drugs .'),
    ('prep_on_account_of',
     '[States,sex] In nineteen-seventy-two Congress approved the Equal Rights Amendment , which stated that `` equality of rights under the law shall not be denied or abridged by the United States or any state on account of sex .'),
    ('prep_on_behalf_of',
     '[responsibility,listener] It is of great consequence in medicine to consider all words , postures , and data , with an underlying responsibility on behalf of the listener to gather the message completely , probing further if there are required clarifications .'),
    ('prep_on_top_of',
     '[placing,roof] The author creates the action packed drama by placing the soldier on top of a roof where he had spent the whole day and into the night on a roof top waiting for the enemy to make a move .'),
    ('prep_onto',
     "[passed,Jem] This moral standing is passed onto Jem and Scout when Atticus says `` Kill all the Blue jays you want , but remember it 's a sin to kill a Mockingbird '' ."),
    ('prep_out',
     '[singling,Boo] Scout singling out Boo is important because it is part of the novels narrative which is that the events are seen from the innocent , non biased point of view -LRB- Scout -RRB- .'),
    ('prep_out_of',
     "[not,something] Esperanza contemplates , `` In Spanish my name is made out of a softer something , like silver , not quite as thick as sister 's name-Magdalena - which is uglier than mine ."),
    ('prep_outside',
     '[sex,marriage] For example , the sex outside the marriage , in these days of easily available contraception , is no more the taboo .'),
    ('prep_outside_of',
     "[interest,occupation] A hobby is an activity one develops to pursue an interest , outside of one 's regular occupation and engages in them primarily for pleasure ."),
    ('prep_over', '[roof,head] I had rather have a small roof over my head than none at all .'),
    ('prep_past',
     '[getting,ways] The way that Atticus tells Jem why they said Tom was guilty just reinforces what Mr. Raymond said about the older people of the town not getting past the ways they were brought up .'),
    ('prep_per',
     '[spend,month] Your credit score along with your available income will determine what kind of interest rate you will have and how much money you can spend per month on a house payment .'),
    ('prep_previous_to',
     "[part,sentence] It seems these changes are mainly due to feeling more appreciated and more as though she is part of the community '' previous to her prison sentence she felt bored and alone ; now however , she has friends and her days are filled with useful tasks ."),
    ('prep_prior_to',
     "[loyal,epiphany] Prior to their epiphany , Sambo and Quimbo were fiercely loyal to their master , to the extent that the term `` Sambo '' is now a literary allusion for an obedient and non-questioning slave ."),
    ('prep_provided', '[symbolizes,that] Provided that , a dictatorial government symbolizes Jack .'),
    ('prep_regarding',
     '[rules,revenge] Kez will not touch heroin and there are rules regarding revenge and joyriding .'),
    ('prep_regardless_of',
     '[equal,race] Stowe views Tom as a paragon of heroism , whose actions are equal to anyone , regardless of race .'),
    ('prep_route',
     '[Atlantic,and] According to research of Alain Coutte , there is a 60-184 million African be aggrieved by slave trade on Atlantic route and % 41.8 is percentage of the slave trade made by the UK .'),
    ('prep_since', '[grew,birth] They grew up in the same village since birth .'),
    ('prep_starting',
     "[blacks,schools] -RRB- `` carpetbaggers '' from the North -LRB- supposedly their just to make a buck , though in fact MOST of them were quite honest and sacrificed and worked hard to rebuild the South , including the recently freed blacks , starting schools , etc -RRB- and blacks -LRB- esp ."),
    ('prep_subsequent_to',
     "[started,vision] Subsequent to Hans ' vision for his son to become a lawyer , Martin started learning about law in 1505 ."),
    ('prep_such_as',
     "[life,death] '' If , after you buy your home and something happens in your life such as a death or illness or the loss of a job , you may be forced to dip into your savings reserve ."),
    ('prep_symbolizes',
     "[.6,promise] -LRB- para .6 -RRB- '' , while Araby symbolizes his promise with the one he loves -LRB- para .11 -RRB- ."),
    ('prep_than', '[anything,escape] He felt that it was not teaching him anything more than a temporary escape .'),
    ('prep_thanks_to',
     '[was,spirits] There were many exciting things about this book , in the end the most exciting part in the book was when scrooge became a complete different person then he ever was all thanks to the spirits .'),
    ('prep_though',
     '[shows,use] The reason why the title is significant because in one line it demonstrates the depth of the conflict between the people and their country , though the use of style in grammar and vocabulary it essentially shows the theme and tone and helps the reader have an idea of what the book is about .'),
    (
        'prep_through',
        '[layered,spelling] The reader hears the accent and so much is layered through altered spelling .'),
    ('prep_throughout',
     '[Siddhartha,life] Govinda accompanies Siddhartha throughout his young life and symbolically , becomes a shadow to Siddhartha .'),
    ('prep_till', '[work,70] It sounds pretty scary if we will have to work till 70 .'),
    ('prep_to', '[similar,ones] I grew up on estates similar to the ones that Kez describes .'),
    ('prep_together_with',
     '[evolved,growth] From Engels point of view , the male dominance in the society evolved together with growth of private property .'),
    ('prep_toward',
     '[bias,things] The authors of all of these documents share many characteristics and for that reason it is important to see how their views may have affected the way they made their documents and that some may be bias toward some things .'),
    ('prep_towards',
     '[bias,equality] Publically opposing the idea of marriage by homosexuals openly shows bias towards equality of citizens under the law and places heterosexuals and homosexuals on separate planes as human beings .'),
    ('prep_under',
     '[shows,law] Publically opposing the idea of marriage by homosexuals openly shows bias towards equality of citizens under the law and places heterosexuals and homosexuals on separate planes as human beings .'),
    ('prep_underneath',
     '[light,door] Until he started seeing a light underneath his bed-room door , opened it , and all he saw was a man in a green robe that was having a feast .'),
    ('prep_unlike',
     "[knowledge,Waters] It was the knowledge of hate that led to majority of the world 's death in the epic story of `` The Waters of Babylon '' by Stephen Vincent Ben `` t. Next , the short story `` How to Build a Fire , '' by Jack London , unlike `` The Waters of Babylon '' by Stephen Vincent Ben `` t , tells of the fall of a man ."),
    ('prep_until',
     "[was,January] It was n't until January 22nd ; however , that he received official power at the ceremony held in La Paz ."),
    ('prep_up',
     '[taking,space] Cyclists have the privilege to ride on the roads but then take it too far by taking up nearly all the space , the cars honk their horns and then cyclists wonder what they are doing wrong !'),
    ('prep_upon', "[placed,homosexuals] Yet this kind of scrutiny is `` justifiably '' placed upon homosexuals ."),
    ('prep_uses',
     "[powerful,personification] He uses personification that helps the reader to understand the meaning of certain phrases and helps the poet to describe the mood more vividly '' ` the fields quivering ' '' this describes the movement of grass under the siege of wind he also personification when describing the skyline '' '' '' '' the skyline a grimace '' '' '' That is very powerful as it makes the reader think how dreadful the storm was ."),
    ('prep_versus',
     '[realism,debate] The scientific realism versus antirealism debate turns on the relevance of epistemology to metaphysics .'),
    ('prep_via',
     '[measure,increases] Functional MRI is a measurement technique based on ultrafast MR imaging sequences that are sensitive to the physiological changes of cerebral blood flow -LRB- CBF -RRB- and cerebral blood volume -LRB- CBV -RRB- . These allow the researcher to measure changes in brain function typically via increases or decreases in blood oxygenation during the scanning -LRB- 2 -RRB- .'),
    ('prep_vs.',
     '[right,fight] Everyone has become caught up in the right vs. left fight and ensuing name calling so few people are truly paying attention to the children themselves .'),
    ('prep_while',
     "[change,ways] Changing the corporate culture in Alumina 's scenario for instance , could eliminate the lack of trust ; change the focus on the problems while developing ways for opportunities ."),
    ('prep_with',
     '[exchanges,neighbor] She describes her spiral from a five year old Kerrie-Ann having intelligent exchanges with a neighbor , into a playground drug dealing Kez and a horrific revenger .'),
    ('prep_with_regard_to',
     '[many,use] With regard to the use of plastic as a packaging material for food and drinks , the advantages are many .'),
    ('prep_with_respect_to',
     '[cause,smokers] With respect to smokers , one individual smoking is not the cause of inconveniences and additional costs ; the accumulations of all the smokers present the issues .'),
    ('prep_within',
     '[exploration,world] An exploration of friendship and altered states of normality within a broken world of deals and highs .'),
    ('prep_without', '[known,company] Due to this , they both have never known a life without each others company .'),
    ('prep_worth',
     '[remember,life] All Americans remember the tragedy of September 11 , 2001 that happened so sudden , but probably not all understand how much worth the life of one person .'),
    ('prepc_about', '[ideas,prevent] Here are some of my ideas about how to prevent foreclosure .'),
    ('prepc_above', '[is,having] If a Reverend and farmer can embody pride , then no one is above having it .'),
    ('prepc_according_to',
     "[see,Berger] According to Berger '' The way we see things is affected by what we know or what we believe '' and `` We never look at just one thing ; we are always looking at the relation between things and ourselves '' ."),
    ('prepc_across',
     '[getting,one] Text messaging might be an easier and efficient way of getting a message across but it has been one of the top ten reasons why teens are being killed on the road .'),
    ('prepc_after',
     '[expected,beaten] After being beaten and shocked in front of a crowd of jeering and drunk white people the narrator is expected to make the speech , after which he receives a college scholarship and the briefcase .'),
    ('prepc_against',
     '[metric,judge] If the -2.5 slope coefficient truly is a robust finding , it can provide us with a metric against which we can judge success or failure of particular policy actions .'),
    ('prepc_ahead_of',
     '[is,were] They could come into a situation where the class they are entering is much further ahead of where they were in their previous school .'),
    ('prepc_along_with',
     '[embrace,incorporating] Along with incorporating multiple technology strategies in the classroom , teachers must embrace the fact that new trends in technology will continue to change the way we teach .'),
    ('prepc_among',
     '[chimera,gauge] The optimal gauge of the absurd may as well be a literary chimera , or a fanciful literary device by means of which the asker feigns the knowable by delving into the unknown , but it is also the fixedly known starting point for any -LRB- further -RRB- enquiry into the ever less known facts among which one is the optimal gauge of the absurd .'),
    ('prepc_around',
     '[think,hindering] It will always force you to think the other way around hindering you from pursuing a greater and a more productive life .'),
    ('prepc_as', '[viewed,getting] We are viewed by others as getting anything we want .'),
    ('prepc_as_of',
     "[agree,judged] So as of having your morals\\/values judged by society I do n't agree with it at all ."),
    ('prepc_as_to',
     "[inkling,expect] '' Petrified , I went to retrieve a kid form a bus with no inkling as to what to expect ."),
    ('prepc_aside_from', '[have,being] Aside from being on time , you have to present yourself appropriately .'),
    ('prepc_at',
     '[looking,causes] By looking at what obesity causes , in our example it affects your body image , we see why it is an issue and why it needs to be taken care of .'),
    ('prepc_away_from',
     "[get,is] '' Hamlet is calling her a slut and telling her to get away from where she is so she will not have sex with everyone in site ."),
    ('prepc_based_on',
     '[respond,influences] Racism is intentionally and unintentionally the isolation and separation of how people generally respond based on what influences as children they may have been convinced of and from cultural traditions that they have experienced .'),
    ('prepc_because',
     "[evidently,giving] They do n't have any discipline and evidently any respect for nothing or no one because they giving off hatred to their own people and destroying their own community ."),
    ('prepc_because_of',
     '[positive,described] The Golden Carp seems much more positive than the Catholic God because of how the different God was described as in his dream .'),
    ('prepc_before',
     "[purchase,going] '' Make a list of items to purchase before going into a store and stick to the list ."),
    ('prepc_behind',
     '[left,goes] Smoke in the air is easily noticeable ; it is the residue that is left behind that goes unnoticed .'),
    ('prepc_beneath',
     '[is,angels] The chain of being is where god is at the top , beneath god are angels and then kings followed by humans and animals etc.'),
    ('prepc_besides',
     "[have,having] His advice on success in the food God bless the child that has his own service industry is `` besides having good food you must have everything else to go with it '' the cleanliness , the service , and the hospitality ."),
    ('prepc_between',
     "[choice,good] Finally , at the end of the novel in Part Three , Alex is `` cured '' and has reverted back to his previous state of having a choice between being good or evil , thus acquiring that sense of free will once more ."),
    ('prepc_but',
     '[oil,suffering] For example , in Africa there are some countries which have oil but they still suffering for poverty . So beside natural resources what can lead a country to touch the richness ?'),
    ('prepc_by',
     '[see,examining] By examining these values , beliefs , and desires communicated through the advertisements we see the many different ways that advertisers use to persuade their audience .'),
    ('prepc_close_to',
     "[come,killing] The other people in the book think he 's supposed to be the savior of the kingdom , yet he ca n't even come close to killing Grendel and when he does attempt to kill Grendel , Grendel just laughs and teases him ."),
    ('prepc_compared_to',
     "[portrayed,portrayed] He did n't change much from how he was portrayed in Beowulf compared to how he was portrayed in Grendel , but this is because Unferth 's encounter with Grendel consumes him completely into this idea of revenge ."),
    ('prepc_depending_on',
     '[affect,placed] However , depending on where the turbines are placed , they can affect television and radio signals .'),
    ('prepc_despite',
     "[experience,executing] Despite executing Tom 's crucifixion , Sambo and Quimbo experience an epiphany during Tom 's death throes ."),
    ('prepc_due_to',
     "[burglaries,malfunctioning] At first , this seemed like the ideal status-booster for Bury St. Edmunds , but already `` arc '' is falling into disrepair : the once magnificent Debenhams department store , constructed with glass walls and a delicate metal structure , has had nearly all of the windows smashed out , as well as several burglaries due to malfunctioning security cameras ."),
    ('prepc_during',
     "[sang,slavery] '' During slavery many slaves sang songs to express how they felt as well as to communicate messages to one another ."),
    ('prepc_except',
     '[merged,share] Herzog merged the two together by creating a two-storey house with a conservatory attached , except that both elements share the same roofline .'),
    ('prepc_far_from',
     '[is,being] The world within the plays of Agamemnon and The Libation Bearers seems just to the people who live in it because this is how they handle all situations that surface , but from an outsiders view , their world is very far from being just .'),
    ('prepc_for', '[think,spend] There is something to think about for what we should spend money on .'),
    ('prepc_from',
     '[stop,prejudice] This brings out that because of the fact that prejudice comes in both ways it is almost impossible to stop people from being prejudice .'),
    ('prepc_if', "[rewarded,did] '' If you did good things in life that means that you will get rewarded ."),
    ('prepc_immediately',
     '[shows,blaming] -LRB- 21 -RRB- Immediately blaming the government , Crane shows realism by metaphorically allegorizing the physical barrier of the regiment to the strict and rigid rules of the army .'),
    ('prepc_in',
     '[poetry,call] Sculpture is like poetry in that call forth certain feelings , certain emotions that function within our heart .'),
    ('prepc_in_addition_to',
     '[two,risky] The last two , in addition to being exceedingly risky , remain technologically in the realm of science fiction -LRB- 92 -RRB- .'),
    ('prepc_including',
     '[life,losing] Frost grew up in rural New England in the early twentieth century and experienced many hardships in his life including losing his father at the young age of eleven and losing two children at very young ages .'),
    ('prepc_inside',
     '[sitting,playing] He tries to influence his audience that there is more to life than sitting inside playing video games , and watching television while our children grow up with little appreciation for nature and what it means .'),
    ('prepc_instead_of',
     "[eat,going] Here are some saving tips that can help you with a budget : '' You can eat at home instead of going to a restaurant ."),
    ('prepc_into',
     '[deceived,believing] Set during the Renaissance period in Venice and on the island of Cyprus , Othello is deceived by the jealous plotting of his right hand man Iago , into believing that his beloved Desdemona is cheating on him .'),
    ('prepc_land', '[agriculture,use] agriculture -RRB- and land use change -LRB- e.g.'),
    ('prepc_like', '[is,looking] It is like looking out a frosted window and not seeing a clear picture .'),
    ('prepc_next',
     '[generation,raised] It is not surprising that this generation next being raised on these influences attitudes is not flexible when it comes to gender roles , racism , and responsibilities .'),
    ('prepc_of',
     "[robbed,may] They are neglected , Kez is abused by adults and robbed of what some may term a `` normal '' childhood , yet the life that she lives has its own measure of right and wrong ."),
    ('prepc_off',
     '[borrows,Charlie] He borrows money off Charlie his friend and goes home with the money saying he has had a wage which shows he lies to his family .'),
    ('prepc_off_of',
     "[got,mayor] The relationship started off great between the two of them , but things started to change when Joe 's head got big off of him being mayor and all the power he had ."),
    ('prepc_on',
     "[hit,laughing] '' That moment for me is always hit bang on when I 'm outside laughing , making snow angels with the ones I love ."),
    ('prepc_on_top_of', '[has,having] On top of having to be alone , he also has to continue living with the guilt .'),
    ('prepc_out',
     '[worked,durring] For the next 2000 years , the ideas that would effect the Chinese civilization , was worked out durring the HAN Dynasty .'),
    ('prepc_out_of',
     '[woken,sleep] There was a great deal of confusion and people were exhausted and dreamy because they had been woken out of there sleep , and told of the misfortune that had happened to them at what should have been the best and the best time in their lives .'),
    ('prepc_over',
     "[debate,able] However , since the 1960s , the nation 's efforts to gain control health care costs have not had much luck to effect , prompting a debate over what suggestions are actually able to continue to reduce costs ."),
    ('prepc_past',
     '[years,could] Almost 150 years past but Africa could not passed over trauma caused by slave trade .'),
    ('prepc_prior_to',
     '[aware,studying] Prior to studying modern American culture I was aware of my desires to buy things .'),
    ('prepc_regarding',
     "[learned,protecting] But , the reincarnated Napster seemed to have learned its lesson regarding protecting its stakeholder 's rights ."),
    ('prepc_regardless_of',
     '[false,represents] Advertisements have led me to believe that every advertisement is false regardless of what business it represents .'),
    ('prepc_since',
     "[keep,carrying] Secondly , since carrying the phone in one 's pocket leads to exposure to his or her bone marrow , one should simply keep the phone off ."),
    ('prepc_starting', '[she,writing] After her husband died she starting writing to support her family .'),
    ('prepc_such_as',
     '[control,establishing] There are several aspects of internal control such as establishing responsibility , using physical , mechanical and electronic controls , segregation of duties , and independent internal verification .'),
    ('prepc_than',
     '[act,acted] If the role of humans and sharks were to be completely switched , the humans would act no different than how the sharks acted .'),
    ('prepc_through',
     '[made,disguising] Through disguising some of these feelings , she not only made her apprehension more evident to the public , but also stimulated more of a response from society , whose reaction appears to have negatively impacted Sexton and her decisions .'),
    ('prepc_throughout',
     '[felt,doing] Throughout doing this assignment , I felt most of times we just ignore our gender role in our society .'),
    ('prepc_to', '[alluding,being] 359 -RRB- , alluding to Tom being on a cross .'),
    ('prepc_toward',
     '[take,believed] The fabricated lies spread by people caused many to take barbaric steps toward what they believed necessary control of wolves .'),
    ('prepc_towards',
     '[feel,helping] The reader also knows that facts are true so if they are true , then all the bad things being said about poverty and how everyone must help must be true , the reader will feel emotions towards helping the charity as poverty seems more realistic and because there are facts and statistics about poverty it has to be real thus putting the situation into perspective .'),
    ('prepc_under',
     '[be,found] The code should also highlight the consequences that all those who were subject to it would be under where they were found to have breached the policy .'),
    ('prepc_unlike',
     "[requires,carving] Unlike carving , it requires soft substances that can be easily and rapidly shaped by the sculptor 's hands ."),
    ('prepc_until',
     "[thought,taking] Another aspect of modern American culture that I had n't thought deeply about until taking this course is love ."),
    ('prepc_upon',
     '[hands,carried] Upon being carried away by his tormentors , Tom says the final words of Christ , `` Into thy hands I commend my spirit !'),
    ('prepc_while',
     '[beat,laughing] Alex instantly chooses the path of evil with the free will that he encompasses , and along with his droogs they beat the old man while laughing at his misery .'),
    ('prepc_with',
     '[faces,adjusting] Her name also seems to remind her of the troubles her family faces with adjusting to life in America .'),
    ('prepc_within',
     '[mean,grow] By space , I mean enough fictional width and imaginary breadth within which the primal absurdness of the literature of the absurd can grow to the most mature phases of the fully developed absurd of the literature of the absurd .'),
    ('prepc_without',
     '[killed,meaning] Hamlet killed Polonius without meaning to , thinking he was killing the king who murdered his father .'),

]

GR_HIERARCHY = \
    (
        "dep",
        (
            "pred",  # Not sure if should be here?
            "aux",
            ("auxpass",
             "cop",
             ),
            "arg",
            ("agent",
             "comp",
             ("acomp",
              "attr",
              "ccomp",
              "xcomp",
              "complm",  # Original Document misstype (compl)
              "pcomp",  # Not sure if should be here?
              "obj",
              ("dobj",
               "iobj",
               "pobj",
               ),
              "mark",
              "rel",
              ),
             "subj",
             ("nsubj",
              ("nsubjpass",
               ),
              "csubj",
              ("csubjpass",
               ),
              ),
             ),
            "cc",
            "conj",
            ("conj_and",
             "conj_negcc",
             'conj_+',
             'conj_and\\/or',
             'conj_as',
             'conj_but',
             'conj_et',
             'conj_if',
             'conj_just',
             'conj_loved',
             'conj_merely',
             'conj_nor',
             'conj_only',
             'conj_or',
             'conj_so',
             'conj_times',
             'conj_v.',
             'conj_versus',
             'conj_vs.',
             'conj_whether',
             'conj_yet',
             "conj_+",
             "conj_negcc",
             ),
            "expl",
            "mod",
            ("abbrev",
             "amod",
             "appos",
             "advcl",
             "purpcl",
             "det",
             "predet",
             "preconj",
             "infmod",
             "partmod",
             "advmod",
             "neg",
             "rcmod",
             "quantmod",
             "tmod",
             "measure",
             "nn",
             "num",
             "number",
             "prep",
             (
                 "prepc",  # Not sure if should be here?
                 "prep_#",
                 "prep_694",
                 "prep_aboard",
                 "prep_about",
                 "prep_above",
                 "prep_according_to",
                 "prep_across",
                 "prep_across_from",
                 "prep_after",
                 "prep_against",
                 "prep_ahead_of",
                 "prep_along",
                 "prep_along_with",
                 "prep_alongside",
                 "prep_although",
                 "prep_amendment",
                 "prep_amid",
                 "prep_among",
                 "prep_amongst",
                 "prep_apart_from",
                 "prep_around",
                 "prep_as",
                 "prep_as_for",
                 "prep_as_from",
                 "prep_as_of",
                 "prep_as_per",
                 "prep_as_to",
                 "prep_aside_from",
                 "prep_at",
                 "prep_away_from",
                 "prep_barring",
                 "prep_based_on",
                 "prep_because",
                 "prep_because_of",
                 "prep_before",
                 "prep_beginning",
                 "prep_behind",
                 "prep_below",
                 "prep_beneath",
                 "prep_beside",
                 "prep_besides",
                 "prep_between",
                 "prep_beyond",
                 "prep_boys",
                 "prep_but",
                 "prep_by",
                 "prep_by_means_of",
                 "prep_close_to",
                 "prep_compared_to",
                 "prep_compared_with",
                 "prep_concerning",
                 "prep_considering",
                 "prep_contrary_to",
                 "prep_depending",
                 "prep_depending_on",
                 "prep_despite",
                 "prep_down",
                 "prep_due_to",
                 "prep_during",
                 "prep_en",
                 "prep_except",
                 "prep_except_for",
                 "prep_far_from",
                 "prep_followed_by",
                 "prep_following",
                 "prep_for",
                 "prep_forth",
                 "prep_from",
                 "prep_genetic",
                 "prep_half",
                 "prep_if",
                 "prep_in",
                 "prep_in_accordance_with",
                 "prep_in_addition_to",
                 "prep_in_case_of",
                 "prep_in_front_of",
                 "prep_in_lieu_of",
                 "prep_in_place_of",
                 "prep_in_spite_of",
                 "prep_including",
                 "prep_inside",
                 "prep_inside_of",
                 "prep_instead_of",
                 "prep_into",
                 "prep_involving",
                 "prep_like",
                 "prep_near",
                 "prep_next",
                 "prep_next_to",
                 "prep_not",
                 "prep_notwithstanding",
                 "prep_of",
                 "prep_off",
                 "prep_off_of",
                 "prep_on",
                 "prep_on_account_of",
                 "prep_on_behalf_of",
                 "prep_on_top_of",
                 "prep_onto",
                 "prep_out",
                 "prep_out_of",
                 "prep_outside",
                 "prep_outside_of",
                 "prep_over",
                 "prep_past",
                 "prep_per",
                 "prep_previous_to",
                 "prep_prior_to",
                 "prep_provided",
                 "prep_regarding",
                 "prep_regardless_of",
                 "prep_route",
                 "prep_since",
                 "prep_starting",
                 "prep_subsequent_to",
                 "prep_such_as",
                 "prep_symbolizes",
                 "prep_than",
                 "prep_thanks_to",
                 "prep_though",
                 "prep_through",
                 "prep_throughout",
                 "prep_till",
                 "prep_to",
                 "prep_together_with",
                 "prep_toward",
                 "prep_towards",
                 "prep_under",
                 "prep_underneath",
                 "prep_unlike",
                 "prep_until",
                 "prep_up",
                 "prep_upon",
                 "prep_uses",
                 "prep_versus",
                 "prep_via",
                 "prep_vs.",
                 "prep_while",
                 "prep_with",
                 "prep_with_regard_to",
                 "prep_with_respect_to",
                 "prep_within",
                 "prep_without",
                 "prep_worth",
                 "prepc_about",
                 "prepc_above",
                 "prepc_according_to",
                 "prepc_across",
                 "prepc_after",
                 "prepc_against",
                 "prepc_ahead_of",
                 "prepc_along_with",
                 "prepc_among",
                 "prepc_around",
                 "prepc_as",
                 "prepc_as_of",
                 "prepc_as_to",
                 "prepc_aside_from",
                 "prepc_at",
                 "prepc_away_from",
                 "prepc_based_on",
                 "prepc_because",
                 "prepc_because_of",
                 "prepc_before",
                 "prepc_behind",
                 "prepc_beneath",
                 "prepc_besides",
                 "prepc_between",
                 "prepc_but",
                 "prepc_by",
                 "prepc_close_to",
                 "prepc_compared_to",
                 "prepc_depending_on",
                 "prepc_despite",
                 "prepc_due_to",
                 "prepc_during",
                 "prepc_except",
                 "prepc_far_from",
                 "prepc_for",
                 "prepc_from",
                 "prepc_if",
                 "prepc_immediately",
                 "prepc_in",
                 "prepc_in_addition_to",
                 "prepc_including",
                 "prepc_inside",
                 "prepc_instead_of",
                 "prepc_into",
                 "prepc_land",
                 "prepc_like",
                 "prepc_next",
                 "prepc_of",
                 "prepc_off",
                 "prepc_off_of",
                 "prepc_on",
                 "prepc_on_top_of",
                 "prepc_out",
                 "prepc_out_of",
                 "prepc_over",
                 "prepc_past",
                 "prepc_prior_to",
                 "prepc_regarding",
                 "prepc_regardless_of",
                 "prepc_since",
                 "prepc_starting",
                 "prepc_such_as",
                 "prepc_than",
                 "prepc_through",
                 "prepc_throughout",
                 "prepc_to",
                 "prepc_toward",
                 "prepc_towards",
                 "prepc_under",
                 "prepc_unlike",
                 "prepc_until",
                 "prepc_upon",
                 "prepc_while",
                 "prepc_with",
                 "prepc_within",
                 "prepc_without",
             ),
             "poss",
             "possessive",
             "prt",
             ),
            "parataxis",
            "punct",
            "ref",
            "sdep",
            ("xsubj",
             )
        )
    )

# The list of tags that represents words, not punctuation - Stanford Adjusted
STANFORD_WORD_TAGS = {
    'NNPS', 'VBP', 'PDT', 'RBS', 'PRP$', 'POS', 'FW', 'ABL', 'ABN', 'ABX', 'AP', 'AT', 'BE', 'BED', 'BEDZ', 'BEG',
    'BEM',
    'BEN', 'BER', 'BEZ', 'CC', 'CD', 'CS', 'DO', 'DOD', 'DOZ', 'DT', 'DTI', 'DTS', 'DTX', 'EX', 'HV', 'HVD', 'HVG',
    'HVN',
    'HVZ', 'IN', 'JJ', 'JJR', 'JJS', 'JJT', 'MD', 'NN', 'NN$', 'NNP', 'NNS', 'NNS$', 'NP', 'NP$', 'NPS', 'NPS$', 'NR',
    'NRS', 'OD', 'PN', 'PN$', 'PP$', 'PP$$', 'PPL', 'PPLS', 'PPO', 'PPS', 'PPSS', 'PRP', 'QL', 'QLP', 'RB', 'RBR',
    'RBT',
    'RN', 'RP', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'WDT', 'WP', 'WP$', 'WPO', 'WPS', 'WQL', 'WRB'}

STEM_EXCEPTIONS = {
    'most':    'many',
    'more':    'many',
    'seconds': 'second',
}


class TagType(str):
    def __new__(cls, tagType: Optional[Union['WNType', str]], wordNetType: Optional[str] = None):
        obj = super(TagType, cls).__new__(cls, str(tagType))
        return obj
    
    def __init__(self, tagType: Optional[Union['WNType', str]], wordNetType: Optional[str] = None):
        assert isinstance(wordNetType, (WNType, type(None)))
        self.wordNetType = wordNetType


class DepType(str):
    @staticmethod
    def fromString(type):
        """
        :type type: str
        :rtype: DepType
        """
        return CDepConst._fromString(type)
    
    def __new__(cls, type):
        """
        :type type: str
        """
        return super(DepType, cls).__new__(cls, str(type))
    
    def __init__(self, type):
        """
        :param type:
        :type type: str
        """
        s = {type, }
        if type in H_DEPENDENCY_EQUIVALENTS:
            s |= H_DEPENDENCY_EQUIVALENTS[type]
        self._equivalents = frozenset(s)
    
    @property
    def equivalent_names(self):
        """
        :rtype : frozenset
        """
        return self._equivalents
    
    @property
    def equivalent_types(self):
        """
        :rtype : frozenset
        """
        return frozenset(map(DepType.fromString, self._equivalents))
    
    def getGeneralType(self):
        '''
        Returns
        :rtype : DepType
        '''
        return getGeneralDependency(self)


class AgentDep(DepType):
    _equivalents = frozenset(('agent', 'nsubj', 'xsubj'))
    
    def __getnewargs__(self, *args, **kwargs):
        """This override is added to enable proper unpickling of the object"""
        return tuple()
    
    def __new__(cls):
        return DepType.__new__(cls, "agent")
    
    def __init__(self):
        pass  # needed to override super init with arguments


class XsubjDep(DepType):
    _equivalents = frozenset(('agent', 'nsubj', 'xsubj'))
    
    def __getnewargs__(self, *args, **kwargs):
        """This override is added to enable proper unpickling of the object"""
        return tuple()
    
    def __new__(cls, *args, **kwargs):
        assert not args and not kwargs
        return DepType.__new__(cls, "xsubj")
    
    def __init__(self):
        pass  # needed to override super init with arguments


class NsubjDep(DepType):
    _equivalents = frozenset(('agent', 'nsubj', 'xsubj'))
    
    def __getnewargs__(self, *args, **kwargs):
        """This override is added to enable proper unpickling of the object"""
        return tuple()
    
    def __new__(cls, *args, **kwargs):
        assert not args and not kwargs
        return DepType.__new__(cls, "nsubj")
    
    def __init__(self):
        pass  # needed to override super init with arguments


DEPENDENCY_EQUIVALENTS = (
    {'nsubjpass', 'dobj', 'partmod', 'rcmod'},  # partmod is with condition that the slave is Noun and not verb
    {'ccomp', 'csubjpass', },
    {'agent', 'nsubj', 'xsubj'},
    {'csubj', 'ccomp', },
    {'partmod', 'infmod', 'vmod'}
)
H_DEPENDENCY_EQUIVALENTS = {d: s - {d, } for s in DEPENDENCY_EQUIVALENTS for d in s}


# H_DEPENDENCY_EQUIVALENTS = dict()
# for s in DEPENDENCY_EQUIVALENTS:
#     for d in s:
#         H_DEPENDENCY_EQUIVALENTS[d] = s - { d, }
# ##for


class CDepConst:
    """
    Contains commonly used Dep names
    """
    _H_BY_NAME = None
    
    @classmethod
    def _fromString(cls, dep):
        '''
        :rtype : DepType
        '''
        if cls._H_BY_NAME is None:
            cls._H_BY_NAME = {str(t[1]): t[1] for t in cls.__dict__.items() if not t[0].startswith('_')}
        return cls._H_BY_NAME.get(dep, DepType(dep))
    
    PRED = DepType('pred')
    AUX = DepType('aux')
    AUXPASS = DepType('auxpass')
    COP = DepType('cop')
    CONJ = DepType('conj')
    CC = DepType('cc')
    PUNCT = DepType('punct')
    ARG = DepType('arg')
    XSUBJ = XsubjDep()
    AGENT = AgentDep()
    SUBJ = DepType('subj')
    NSUBJ = NsubjDep()
    NSUBJPASS = DepType('nsubjpass')
    CSUBJ = DepType('csubj')
    CSUBJPASS = DepType('csubjpass')
    COMP = DepType('comp')
    OBJ = DepType('obj')
    DOBJ = DepType('dobj')
    IOBJ = DepType('iobj')
    POBJ = DepType('pobj')
    PCOMP = DepType('pcomp')
    ATTR = DepType('attr')
    CCOMP = DepType('ccomp')
    XCOMP = DepType('xcomp')
    COMPLM = DepType('complm')
    MARK = DepType('mark')
    REL = DepType('rel')
    REF = DepType('ref')
    EXPL = DepType('expl')
    ACOMP = DepType('acomp')
    MOD = DepType('mod')
    ADVCL = DepType('advcl')
    PURPCL = DepType('purpcl')
    TMOD = DepType('tmod')
    RCMOD = DepType('rcmod')
    AMOD = DepType('amod')
    NUM = DepType('num')
    NUMBER = DepType('number')
    QUANTMOD = DepType('quantmod')
    NN = DepType('nn')
    APPOS = DepType('appos')
    ABBREV = DepType('abbrev')
    PARTMOD = DepType('partmod')
    INFMOD = DepType('infmod')
    VMOD = DepType('vmod')  # virtual dependency, reduced, non-finite verbal modifier (infmod|partmod)
    ADVMOD = DepType('advmod')
    NEG = DepType('neg')
    MEASURE = DepType('measure')
    DET = DepType('det')
    PREDET = DepType('predet')
    PRECONJ = DepType('preconj')
    POSS = DepType('poss')
    POSSESSIVE = DepType('possessive')
    PREP = DepType('prep')
    PREPC = DepType('prepc')
    PREP_WITH = DepType('prep_with')
    PRT = DepType('prt')
    PARATAXIS = DepType('parataxis')
    SDEP = DepType('sdep')


__H_GENERAL_DEPENDENCIES = {
    'nsubjpass': CDepConst.DOBJ,
    'nsubj':     CDepConst.AGENT,
    'xsubj':     CDepConst.AGENT,
    'csubjpass': CDepConst.CCOMP,
    'csubj':     CDepConst.CCOMP,
    'infmod':    CDepConst.VMOD,
    'partmod':   CDepConst.VMOD
}


def getGeneralDependency(dep):
    """
    :type dep: basestring | DepType
    :rtype: DepType
    """
    if dep in __H_GENERAL_DEPENDENCIES:
        return __H_GENERAL_DEPENDENCIES[dep]
    elif isinstance(dep, DepType):
        return dep
    else:
        return DepType.fromString(dep)


# class CDepConst ~~~~~~16.02.2013 15:36:03~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def recurs_GR_hier(tree):
    """
    Creates and returns a dictionary wich is used for fast finding all descendants for every dependency in the Hierarchy
    """
    hMainGRs = {}
    for i, gr in enumerate(tree):
        if isinstance(gr, tuple):
            hR = recurs_GR_hier(gr)
            ##Adding all children
            for v in hR.values():
                hMainGRs[tree[i - 1]] |= v
            hMainGRs.update(hR)
        else:
            hMainGRs[gr] = {gr, }
    return hMainGRs


H_GRAMMATICAL_RELATION_PARSED_HIERARCHY = recurs_GR_hier(GR_HIERARCHY)


class WNType(str):
    def __new__(cls, type: str):
        return super(WNType, cls).__new__(cls, str(type))


class WordNetConsts:
    NOUN = WNType('n')
    VERB = WNType('v')
    ADJECTIVE = WNType('a')
    ADJECTIVE_SATELLITE = WNType('s')
    ADVERB = WNType('r')


class CTags:
    '''
    List of Penn TreeBank POS tags
    '''
    
    @classmethod
    def fromString(cls, s):
        d = {str(v): v for v in cls.__dict__.values() if isinstance(v, TagType)}
        if s in d:
            return d[s]
        else:
            return TagType(s)
    
    @classmethod
    def isWordType(cls, tag: str):
        d = {str(v) for v in cls.__dict__.values() if isinstance(v, TagType)}
        return tag == tag.upper() and str(tag) in d - {'CD', 'LS', 'SYM', 'UH'}
        # return tag == tag.upper() and tag in set(cls.__dict__.keys()) - {'CD', 'LS', 'SYM', 'UH'}
    
    """Coordinating conjunction"""
    CC = TagType("CC")
    """Cardinal number"""
    CD = TagType("CD")
    """Determiner"""
    DT = TagType("DT")
    """Existential there"""
    EX = TagType("EX")
    """Foreign word"""
    FW = TagType("FW")
    """Preposition or subordinating conjunction"""
    IN = TagType("IN")
    """Adjective"""
    JJ = TagType("JJ", WordNetConsts.ADJECTIVE)
    """Adjective, comparative"""
    JJR = TagType("JJR", WordNetConsts.ADJECTIVE)
    """Adjective, superlative"""
    JJS = TagType("JJS", WordNetConsts.ADJECTIVE)
    """List item marker"""
    LS = TagType("LS")
    """Modal"""
    MD = TagType("MD", WordNetConsts.VERB)
    """Noun, singular or mass"""
    NN = TagType("NN", WordNetConsts.NOUN)
    """Noun, plural"""
    NNS = TagType("NNS", WordNetConsts.NOUN)
    """Proper noun, singular"""
    NNP = TagType("NNP", WordNetConsts.NOUN)
    """Proper noun, plural"""
    NNPS = TagType("NNPS", WordNetConsts.NOUN)
    """Predeterminer"""
    PDT = TagType("PDT")
    """Possessive ending"""
    POS = TagType("POS")
    """Personal pronoun"""
    PRP = TagType("PRP")
    """Possessive pronoun"""
    PPRP = TagType("PRP$")
    """Adverb"""
    RB = TagType("RB", WordNetConsts.ADVERB)
    """Adverb, comparative"""
    RBR = TagType("RBR", WordNetConsts.ADVERB)
    """Adverb, superlative"""
    RBS = TagType("RBS", WordNetConsts.ADVERB)
    """Particle"""
    RP = TagType("RP")
    """Symbol"""
    SYM = TagType("SYM")
    """to"""
    TO = TagType("TO")
    """Interjection"""
    UH = TagType("UH")
    """Verb, base form"""
    VB = TagType("VB", WordNetConsts.VERB)
    """Verb, past tense"""
    VBD = TagType("VBD", WordNetConsts.VERB)
    """Verb, gerund or present participle"""
    VBG = TagType("VBG", WordNetConsts.VERB)
    """Verb, past participle"""
    VBN = TagType("VBN", WordNetConsts.VERB)
    """Verb, non-3rd person singular present"""
    VBP = TagType("VBP", WordNetConsts.VERB)
    """Verb, 3rd person singular present"""
    VBZ = TagType("VBZ", WordNetConsts.VERB)
    """Wh-determiner"""
    WDT = TagType("WDT")
    """Wh-pronoun"""
    WP = TagType("WP")
    """Possessive wh-pronoun"""
    PWP = TagType("WP$")
    """Wh-adverb"""
    WRB = TagType("WRB")


class VerbFrameType(int):
    def __new__(cls, idn, descr, **kwrds):
        """
        :type type: int
        :type descr: str
        """
        obj = super(VerbFrameType, cls).__new__(cls, idn)
        obj.description = descr  # public
        return obj


class VerbFrames(object):
    @staticmethod
    def fromInt(i):
        return VerbFramesTuples[i][2]
    
    VF_0 = VerbFrameType(0, None)
    VF_1 = VerbFrameType(1, "Something %s")
    VF_2 = VerbFrameType(2, "Somebody %s")
    VF_3 = VerbFrameType(3, "It is %sing")
    VF_4 = VerbFrameType(4, "Something is %sing PP")
    VF_5 = VerbFrameType(5, "Something %s something Adjective/Noun")
    VF_6 = VerbFrameType(6, "Something %s Adjective/Noun")
    VF_7 = VerbFrameType(7, "Somebody %s Adjective")
    VF_8 = VerbFrameType(8, "Somebody %s something")
    VF_9 = VerbFrameType(9, "Somebody %s somebody")
    VF_10 = VerbFrameType(10, "Something %s somebody")
    VF_11 = VerbFrameType(11, "Something %s something")
    VF_12 = VerbFrameType(12, "Something %s to somebody")
    VF_13 = VerbFrameType(13, "Somebody %s on something")
    VF_14 = VerbFrameType(14, "Somebody %s somebody something")
    VF_15 = VerbFrameType(15, "Somebody %s something to somebody")
    VF_16 = VerbFrameType(16, "Somebody %s something from somebody")
    VF_17 = VerbFrameType(17, "Somebody %s somebody with something")
    VF_18 = VerbFrameType(18, "Somebody %s somebody of something")
    VF_19 = VerbFrameType(19, "Somebody %s something on somebody")
    VF_20 = VerbFrameType(20, "Somebody %s somebody PP")
    VF_21 = VerbFrameType(21, "Somebody %s something PP")
    VF_22 = VerbFrameType(22, "Somebody %s PP")
    VF_23 = VerbFrameType(23, "Somebody's (body part) %s")
    VF_24 = VerbFrameType(24, "Somebody %s somebody to INFINITIVE")
    VF_25 = VerbFrameType(25, "Somebody %s somebody INFINITIVE")
    VF_26 = VerbFrameType(26, "Somebody %s that CLAUSE")
    VF_27 = VerbFrameType(27, "Somebody %s to somebody")
    VF_28 = VerbFrameType(28, "Somebody %s to INFINITIVE")
    VF_29 = VerbFrameType(29, "Somebody %s whether INFINITIVE")
    VF_30 = VerbFrameType(30, "Somebody %s somebody into V-ing something")
    VF_31 = VerbFrameType(31, "Somebody %s something with something")
    VF_32 = VerbFrameType(32, "Somebody %s INFINITIVE")
    VF_33 = VerbFrameType(33, "Somebody %s VERB-ing")
    VF_34 = VerbFrameType(34, "It %s that CLAUSE")
    VF_35 = VerbFrameType(35, "Something %s INFINITIVE")


VerbFramesTuples = [
    (0, None, VerbFrames.VF_0),
    (1, "Something %s", VerbFrames.VF_1),
    (2, "Somebody %s", VerbFrames.VF_2),
    (3, "It is %sing", VerbFrames.VF_3),
    (4, "Something is %sing PP", VerbFrames.VF_4),
    (5, "Something %s something Adjective/Noun", VerbFrames.VF_5),
    (6, "Something %s Adjective/Noun", VerbFrames.VF_6),
    (7, "Somebody %s Adjective", VerbFrames.VF_7),
    (8, "Somebody %s something", VerbFrames.VF_8),
    (9, "Somebody %s somebody", VerbFrames.VF_9),
    (10, "Something %s somebody", VerbFrames.VF_10),
    (11, "Something %s something", VerbFrames.VF_11),
    (12, "Something %s to somebody", VerbFrames.VF_12),
    (13, "Somebody %s on something", VerbFrames.VF_13),
    (14, "Somebody %s somebody something", VerbFrames.VF_14),
    (15, "Somebody %s something to somebody", VerbFrames.VF_15),
    (16, "Somebody %s something from somebody", VerbFrames.VF_16),
    (17, "Somebody %s somebody with something", VerbFrames.VF_17),
    (18, "Somebody %s somebody of something", VerbFrames.VF_18),
    (19, "Somebody %s something on somebody", VerbFrames.VF_19),
    (20, "Somebody %s somebody PP", VerbFrames.VF_20),
    (21, "Somebody %s something PP", VerbFrames.VF_21),
    (22, "Somebody %s PP", VerbFrames.VF_22),
    (23, "Somebody's (body part) %s", VerbFrames.VF_23),
    (24, "Somebody %s somebody to INFINITIVE", VerbFrames.VF_24),
    (25, "Somebody %s somebody INFINITIVE", VerbFrames.VF_25),
    (26, "Somebody %s that CLAUSE", VerbFrames.VF_26),
    (27, "Somebody %s to somebody", VerbFrames.VF_27),
    (28, "Somebody %s to INFINITIVE", VerbFrames.VF_28),
    (29, "Somebody %s whether INFINITIVE", VerbFrames.VF_29),
    (30, "Somebody %s somebody into V-ing something", VerbFrames.VF_30),
    (31, "Somebody %s something with something", VerbFrames.VF_31),
    (32, "Somebody %s INFINITIVE", VerbFrames.VF_32),
    (33, "Somebody %s VERB-ing", VerbFrames.VF_33),
    (34, "It %s that CLAUSE", VerbFrames.VF_34),
    (35, "Something %s INFINITIVE", VerbFrames.VF_35)
]
