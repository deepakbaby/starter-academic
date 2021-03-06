+++
title = "ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസ് സീരീസ്: ഭാഗം - 4 "
subtitle = ""
date = 2018-04-24
draft = false

authors = ["admin"]

tags = ["ai", "malayalam"]
categories = []

math = true

[image]
  caption = ""
  focal_point = "Smart"

+++

ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസ് ഗവേഷണത്തിൽ ഇന്ന് ഏറ്റവുമധികം ഉപയോഗത്തിലിരിക്കുന്ന ന്യൂറൽ നെറ്റ്‌വർക്കുകളെ കുറിച്ചാണ് കഴിഞ്ഞ ഭാഗത്തിൽ പറഞ്ഞത് (കഴിഞ്ഞ ഭാഗം [ഇവിടെ](https://dblogsai.blogspot.be/2018/05/3.html) വായിക്കാം). ഇനി ചരിത്രത്തിലെ രണ്ടാം ഘട്ടത്തിലേക്ക്. മുമ്പുപറഞ്ഞതുപോലെ ന്യൂറൽ നെറ്റ്‌വർക്കുമായി ബന്ധപ്പെട്ട ആദ്യകാല ശ്രമങ്ങളും ഗവേഷണങ്ങളുമാണ് ഈ ഭാഗത്തിൽ.  
  
**1943 - ഇലെക്ട്രിക്കൽ സർക്യൂട്ടുകൾ ഉപയോഗിച്ച് ഒരു ന്യൂറൽ നെറ്റ്‌വർക്ക് നിർമ്മിക്കപ്പെട്ടു**  
നമ്മുടെ ന്യൂറോണുകൾ എങ്ങനെയായിരിക്കും പ്രവർത്തിക്കുന്നത് എന്നതിനെപ്പറ്റി ന്യൂറോഫൈസിയോളജിസ്റ്റായ വാറൻ മക്കുല്ലോഷും ഗണിതജ്ഞനായ വാൾട്ടർ പിട്സും ചേർന്ന് ഒരു സിദ്ധാന്തം അവതരിപ്പിച്ചു. അതിന്റെ ഒരു ചെറിയ മോഡൽ ഇലെക്ട്രിക്കൽ സർക്യൂട്ടുകൾ ഉപയോഗിച്ച് അവർ നിർമിക്കുകയും ചെയ്തു. നമ്മുടെ ശരീരത്തിലെ ന്യൂറോണുകളുടെ പ്രവർത്തനത്തെപ്പറ്റി അനുലഭ്യമായിരുന്ന പരിമിതമായ അറിവുവച്ചാണ് അത്തരമൊരു മാതൃക അവർ നിർമ്മിച്ചത്.  
**1952- checkers game കളിക്കുന്ന കമ്പ്യൂട്ടർ**  

[![](https://3.bp.blogspot.com/-FaFFCwAsMbQ/WvF5hNzBFkI/AAAAAAAACqo/8f70fHH691w5Uz83QVXCaSzCp49nvh6zACEwYBhgL/s320/checkers%2Bgame.jpg)](https://3.bp.blogspot.com/-FaFFCwAsMbQ/WvF5hNzBFkI/AAAAAAAACqo/8f70fHH691w5Uz83QVXCaSzCp49nvh6zACEwYBhgL/s1600/checkers%2Bgame.jpg)

അമേരിക്കയിൽ പ്രചാരത്തിലുള്ള ഒരു ബോർഡ് ഗെയിം ആണ് checkers. AI യുടെ തുടക്കക്കാരിലൊരാളായ [ആർതർ സാമുവേൽ](https://en.wikipedia.org/wiki/Arthur_Samuel) ഓരോ കളികഴിയുന്തോറും കളി മെച്ചപ്പെടുത്തുന്ന ഒരു കമ്പ്യൂട്ടർ പ്രോഗ്രാം വികസിപ്പിച്ചു. ആരംഭകാലത്തെ AI ഗവേഷണങ്ങൾ മിക്കതും വികസിപ്പിക്കപ്പെട്ടത് ബോർഡ് ഗെയിമുകളെ അടിസ്ഥാനപ്പെടുത്തിയായിരുന്നു. താരതമ്യേന ലളിതമായ നിയമങ്ങളും എന്നാൽ കളിക്കുമ്പോൾ വളരെയധികം സങ്കീര്ണമാവുകയും ചെയ്യുന്നവയാണ് മിക്ക ബോർഡ് ഗെയിമുകളും (ചെസ്സ് ആണ് ഒരു നല്ല ഉദാഹരണം). ആർതർ സാമുവേൽ ആണ് മെഷീൻ ലേർണിംഗ് എന്ന വാക്ക് ആദ്യമായി അവതരിപ്പിച്ചത്.  
  
**1957- The perceptron**  
1957-ൽ ഫ്രാങ്ക് റോസെൻബ്ലേറ്റ് എന്ന ഗവേഷകൻ [പെർസെപ്ട്രോൺ](https://en.wikipedia.org/wiki/Perceptron) എന്ന ഒരുതരം ന്യൂറൽ നെറ്റ്‌വർക്ക് അവതരിപ്പിച്ചു. നമ്മുടെ തലച്ചോറിലെയെന്നപോലെ പല ന്യൂറോണുകളിൽനിന്നും വരുന്ന വിവരങ്ങളെ എകോപിപ്പിച്ചു തീരുമാനങ്ങൾ എടുക്കാൻ സാധിക്കത്തക്കരീതിയിലാണ് ഇവയെ രൂപകല്പന ചെയ്തിരുന്നത്. ഭാവിയിൽ മനുഷ്യരെപ്പോലെ നടക്കാനും സംസാരിക്കാനും കാണാനുമൊക്കെ കഴിയുകയും സ്വന്തമായി ഒരു അസ്തിത്വമുണ്ടെന്നു സ്വയം മനസിലാക്കാനും കഴിയുന്ന കംപ്യൂട്ടറുകളിലേക്കുള്ള ആദ്യപടിയാണിതെന്നാണ് അദ്ദേഹം ഇതിനെപറ്റി പറഞ്ഞത്. ഇത് അക്കാലത്തു വിവാദമാവുകയും ചെയ്തിരുന്നു. ഒരു ഫോട്ടോയിലുള്ളത് ഒരു വസ്തുവാണോ അല്ലയോ എന്നുള്ള തീരുമാനം മാത്രം എടുക്കാൻ സാധിക്കുന്നവയായിരുന്നു പെർസെപ്ട്രോൺസ്. ഉദാഹരണത്തിന് ഒരു ചിത്രത്തിലുള്ളത് നായയാണോ അല്ലയോ എന്ന് മാത്രം പറയാൻ കഴിയുന്നവ. പൂച്ചയുടെ ചിത്രം കാണിച്ചാൽ അതു പൂച്ചയാണെന്നു അതിനു മനസിലാക്കാനാവില്ല. പകരം അതൊരു നായയല്ല എന്ന് മാത്രം മനസിലാകും (Dog അല്ലെങ്കിൽ Not Dog). ആരംഭകാലത്തു പെർസെപ്ട്രോൺസ് പ്രതീക്ഷക്കു വക നൽകിയിരുന്നെങ്കിലും പിന്നീട് അവക്ക് ഒന്നിലധികം വസ്തുക്കളെ തിരിച്ചറിയാനാകില്ല എന്ന പോരായ്മ മൂലം അധികം ഗവേഷകർ അത് എറ്റെടുത്തില്ല. ന്യൂറൽ നെറ്റ്‌വർക്ക് അടിസ്ഥാനമാക്കിയുള്ള ഗവേഷണം ഇതുകൊണ്ടൊക്കെ കുറെ നാൾ ആരും തുടർന്നില്ല.  
  
**1959- Stanford's MADALINE**  
ഒരു ന്യൂറൽ നെറ്റ്‌വർക്ക് ആദ്യമായി ഒരു റിയൽ വേൾഡ് അപ്പ്ലിക്കേഷനിൽ ഉപയോഗിക്കപ്പെട്ടു. സ്റ്റാൻഫോർഡിൽ വികസിപ്പിക്കപ്പെട്ട [MADALINE](https://en.wikipedia.org/wiki/ADALINE) എന്ന പ്രോഗ്രാം ഫോൺ കോളുകളിലെ മുഴക്കം (echo) കുറയ്ക്കാനാണ് ഉപയോഗിച്ചത്. ഈ പ്രോഗ്രാം ഇന്നും ഉപയോഗിക്കപ്പെടുന്നുണ്ട്. ഇത് പൂർണമായ അർത്ഥത്തിൽ ഒരു ന്യൂറൽ നെറ്റ്‌വർക്ക് എന്ന് പറയാനാവില്ലെങ്കിലും ഗവേഷണങ്ങളിൽ മാത്രം ഒതുങ്ങാതെ നമ്മുടെ ദൈനംദിന ആവശ്യങ്ങൾക്കും ഇത്തരം അൽഗോരിതങ്ങൾ ഉപയോഗപ്പെടുത്താമെന്നതിനുള്ള ഒരുപക്ഷേ ആദ്യത്തെ ഉദാഹരണമാവും MADALINE.  
  
**1985- NETtalk**  
ഇംഗ്ലീഷ് വാക്കുകൾ ഉച്ചരിക്കാൻവേണ്ടി ടെറി സെയ്‌നോവ്സ്കി, ചാൾസ് റോസെൻബെർഗ് എന്നീ ഗവേഷകർ രൂപകല്പന ചെയ്ത ആർട്ടിഫിഷ്യൽ ന്യൂറൽ നെറ്റ്‌വർക്ക് ആണ് [NETtalk](https://en.wikipedia.org/wiki/NETtalk_(artificial_neural_network)). 20000 ഇംഗ്ലീഷ് വാക്കുകൾ ഉച്ചരിക്കുവാൻ ഈ ന്യൂറൽ നെറ്റ്‌വർക്കിന് കഴിയുമായിരുന്നു. അതായത്, അതിലേക്കു ഇംഗ്ലീഷ് വാക്ക് ഇൻപുട്ടായി കൊടുത്താൽ ആ വാക്കിന്റെ ഉച്ചാരണം അത് ഔട്പുട്ടിൽ നൽകും. ഇവ എങ്ങനെയാണ് പ്രവർത്തിച്ചിരുന്നതെന്നു മനസ്സിലാക്കണമെങ്കിൽ നമുക്ക് കൂടുതൽ കാര്യങ്ങൾ പഠിക്കേണ്ടതുണ്ട്. അതിലേക്കു അടുത്ത ഭാഗത്തിൽ കടക്കാം.  
  
കാര്യങ്ങൾ പ്രതീക്ഷക്കു വകനല്കുന്നവ ആയിരുന്നെങ്കിലും ന്യൂറൽ നെറ്റ്‌വർക്ക് അടിസ്ഥാനപ്പെടുത്തിയുള്ള AI ഗവേഷണങ്ങൾ അധികമാരും തുടർന്നില്ല. അതിനുള്ള പ്രധാനകാരണങ്ങൾ ഇവയാണ്.  

1.  ന്യൂറൽ നെറ്റ്‌വർക്കുകൾ എങ്ങനെയാണ് പ്രവർത്തിക്കുന്നതെന്നോ അവയെ എങ്ങനെ ഓരോ കൃത്യങ്ങൾക്കു ഉപയോഗിക്കാമെന്നുള്ള വ്യകതമായ ധാരണയോ പണ്ടുണ്ടായിരുന്നില്ല. അതിൽ ഉപയോഗിയ്ക്കപ്പെടുന്ന സങ്കീർണമായഗണിതവും അത് കൃത്യമായി നിർധാരണം ചെയ്തെടുക്കാനുള്ള രീതികൾ നിലവിലില്ലാതിരുന്നതുമായിരുന്നു കാരണം.
2.  ന്യൂറൽ നെറ്റ്‌വർക്കുകൾക്കു പാറ്റേണുകൾ പഠിക്കണമെങ്കിൽ ധാരാളം ട്രെയിനിങ് ഡാറ്റ ആവശ്യമായിരുന്നു. അന്നത്തെക്കാലത്ത് ഇന്നുള്ളതുപോലെ ബിഗ് ഡാറ്റയൊന്നും ലഭ്യമായിരുന്നില്ല.
3.  ന്യൂറൽ നെറ്റ്‌വർക്കുകൾ ട്രെയിൻ ചെയ്യാൻ, ഓരോവസ്തുക്കളിലെയും പാറ്റേണുകൾ തമ്മിൽ വേർതിരിച്ചറിയുന്നതെങ്ങനെ എന്നതൊക്കെ പഠിപ്പിക്കാൻ ധാരാളം കമ്പ്യൂട്ടർ പവർ വേണമായിരുന്നു. അന്ന് അതും ഉണ്ടായിരുന്നില്ല.
4.  ന്യൂറൽ നെറ്റ്‌വർക്കുകളെ ഓരോ ടാസ്കുകൾ ചെയ്യാൻ പഠിപ്പിക്കുന്ന ട്രെയിനിങ് രീതികൾ അന്നധികം വികസിപ്പിക്കപ്പെട്ടിരുന്നില്ല. ഉള്ളവയാകട്ടെ ധാരാളം സമയമെടുക്കുന്നവയും!

ഇതോടൊപ്പം AI രംഗത്ത് താരതമ്യേന കുറച്ചു കമ്പ്യൂട്ടർ റിസോഴ്‌സ് ആവശ്യമുള്ള [സപ്പോർട് വെക്ടർ മഷീൻസ്](https://en.wikipedia.org/wiki/Support_vector_machine), [nearest neighbor](https://en.wikipedia.org/wiki/Nearest_neighbor_search) തുടങ്ങിയ സങ്കേതങ്ങൾ വ്യാപകമായി ഉപയോഗത്തിൽ വന്നു. അതോടെ ന്യൂറൽ നെറ്റ്‌വർക്ക് ഗവേഷണം എകദേശം എല്ലാവരും കൈവിട്ടു. പിന്നീടൊരു ഉയർത്തെഴുന്നേല്പുണ്ടാകുന്നത് 2006 ഇലാണ്. അതേപറ്റി അടുത്ത ഭാഗം ചരിത്രത്തിൽ.

