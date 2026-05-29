\ProvidesClass{iisreport}

% The 'parskip' option causes paragraphs to start with a vertical space of half a line instead of
% indentation.
\DeclareOption{parskip}{%
 \PassOptionsToClass{parskip=half}{scrbook}
}

% The 'print' option optimizes the page layout for printing and binding the report.
\newif\if@printlayout
\@printlayoutfalse
\DeclareOption{print}{%
 \@printlayouttrue
 \PassOptionsToClass{%
  BCOR=5mm, % Our ring bindings are 5mm wide.
  twoside,
 }{scrbook}
}
\DeclareOption{noprint}{%
 \@printlayoutfalse
 \PassOptionsToClass{oneside}{scrbook}
}

% The 'oldfonts' option brings back the fonts from the legacy IIS report template.
\newif\if@oldfonts
\@oldfontsfalse
\DeclareOption{oldfonts}{%
 \@oldfontstrue
}

% The 'oldsubscript' option disables having a printable underscore in favor of being able to use it
% for subscripts.
\newif\if@oldsubscript
\@oldsubscriptfalse
\DeclareOption{oldsubscript}{%
 \@oldsubscripttrue
}

% Pass unknown options to the base class.
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{scrbook}}
\ExecuteOptions{noprint}
\ProcessOptions\relax

% % Make underscore a printable character.  Use `\sb{}` for subscripts.  (For reasons of symmetry,
% % you should use `\sp{}` for superscripts.)
% \if@oldsubscript
% \else
%  \catcode`_=12
% \fi

% The underlying base class is `scrbook`.
\LoadClass[%
 a4paper,
 listof=totoc,       % Add "List of .." entries to the Table of Contents.
 bibliography=totoc, % Add Bibliography entry to the Table of Contents.
]{scrbook}

% Workaround to LaTeX complaining about the use of deprecated `\rm`.
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}

% % Use UTF-8 to encode input and Type 1 font encoding.
% % \RequirePackage[utf8]{inputenc}
% \RequirePackage[T1]{fontenc}

% % Font packages

% % Kepler fonts for mathematics, Pazo (a Palatino derivative) for text, and Bera Mono for monospace.
% % In contrast to the default (La)TeX fonts, these fonts were designed by experienced typographers.
% \RequirePackage{amssymb}
% \RequirePackage[notext]{kpfonts}
% \RequirePackage{mathpazo}
% \RequirePackage[scaled=.85]{beramono}

% \RequirePackage[numbers]{natbib}

\RequirePackage{iftex}

\ifPDFTeX
  \RequirePackage[T1]{fontenc}
  \RequirePackage[utf8]{inputenc}
  \RequirePackage{amssymb}
  \RequirePackage[notext]{kpfonts}
  \RequirePackage{mathpazo}
  \RequirePackage[scaled=.85]{beramono}
  \RequirePackage{mathtools}
\else
  \RequirePackage{fontspec}

  \setmainfont{TeX Gyre Pagella}
  \setsansfont{TeX Gyre Heros}
  \setmonofont[Scale=MatchLowercase]{DejaVu Sans Mono}

  \RequirePackage{mathtools}
  \RequirePackage[
    math-style=ISO,
    bold-style=ISO,
    warnings-off={mathtools-colon,mathtools-overbracket}
  ]{unicode-math}

  \setmathfont{Latin Modern Math}
\fi

% Use American English spelling and hyphenation.
\RequirePackage[english]{babel}

% Recommended for biblatex; gives language-aware quotation handling.
\RequirePackage{csquotes}
\RequirePackage{xurl}

\RequirePackage{silence}
% Warning is a bug in biblatex that is fixed in the newest upstream and does not affect output
\WarningFilter{biblatex}{Macro 'volume+number' undefined}
% \RequirePackage[
%   backend=biber,
%   style=ieee,
%   sorting=none,
%   natbib=true,
%   giveninits=true,
%   maxnames=6,
%   minnames=1,
%   maxbibnames=6,
%   minbibnames=1,
%   doi=true,
%   url=false,
%   isbn=false,
%   eprint=true,
%   dashed=false
% ]{biblatex}
\RequirePackage[
  backend=biber,
  bibstyle=ieee,
  citestyle=numeric-comp,
  sorting=none,
  sortcites=true,
  natbib=true,
  giveninits=true,
  maxcitenames=1,
  mincitenames=1,
  maxbibnames=6,
  minbibnames=1,
  doi=true,
  url=false,
  isbn=false,
  eprint=true,
  dashed=false
]{biblatex}

\RequirePackage{ragged2e}

\AtBeginBibliography{%
  \small
  \RaggedRight
  % \setlength{\emergencystretch}{3em}
  \Urlmuskip=0mu plus 2mu\relax
}

\AtEveryBibitem{%
  \clearfield{abstract}%
  \clearfield{issn}%
  \clearfield{isbn}%
  \clearlist{language}%
  \clearfield{urlday}%
  \clearfield{urlmonth}%
  \clearfield{urlyear}%
  \clearfield{urldate}%
}

% `graphicx` enables including all kinds of images (including single-page PDFs).  The default search
% path is set to the `fig/` directory.
\RequirePackage{graphicx}
\graphicspath{{fig/}}

% `xcolor` adds rich color support.
\RequirePackage[dvipsnames, svgnames, table]{xcolor}

% Customize chapter headings.
% \RequirePackage[Lenny]{fncychap}
\newlength{\iis@chrulewidth}
\newlength{\iis@chpx}
\newlength{\iis@chpy}
\newlength{\iis@chmylen}
\newlength{\iis@chpyy}
\newlength{\iis@chmyhi}

\newcommand*{\iis@chapterprefixfont}{%
  \normalfont\sffamily\fontsize{14}{16}\selectfont%
}

\newcommand*{\iis@chapternumberfont}{%
  \normalfont\rmfamily\fontsize{60}{62}\selectfont%
}

\newcommand*{\iis@chaptertitlefont}{%
  \normalfont\rmfamily\Huge\bfseries%
}

\newcommand*{\iis@chapterrulefill}[1]{%
  \leavevmode\leaders\hrule height #1\hfill\kern\z@%
}

\setlength{\iis@chrulewidth}{1pt}

\KOMAoptions{chapterprefix=true}

\setkomafont{chapter}{\iis@chaptertitlefont}

\RedeclareSectionCommand[
  beforeskip=-61pt,
  innerskip=0pt,
  afterskip=40pt
]{chapter}

\renewcommand*{\chapterlineswithprefixformat}[3]{%
  \raggedchapter
  \IfArgIsEmpty{#2}{}{%
    % Width/height of "Chapter"
    \settowidth{\iis@chpx}{\iis@chapterprefixfont\chapapp}%
    \addtolength{\iis@chpx}{2pt}%
    \settoheight{\iis@chpy}{\iis@chapterprefixfont\chapapp}%
    \addtolength{\iis@chpy}{1pt}%
    %
    % Width of "Chapter <number>"
    \settowidth{\iis@chmylen}{%
      \iis@chapterprefixfont\chapapp\space
      \iis@chapternumberfont\thechapter%
    }%
    \addtolength{\iis@chmylen}{1pt}%
    %
    % Height of the chapter number
    \settoheight{\iis@chpyy}{%
      \iis@chapternumberfont\thechapter%
    }%
    \addtolength{\iis@chpyy}{-2pt}%
    %
    % Left vertical rule height, deliberately excluding the "Chapter" text zone
    \setlength{\iis@chmyhi}{\iis@chpyy}%
    \addtolength{\iis@chmyhi}{-\iis@chpy}%
    %
    \par\nobreak
    \parbox[b]{\linewidth}{%
      % Left vertical rule, only above the word "Chapter"
      \rule[\iis@chpy]{\iis@chrulewidth}{\iis@chmyhi}%
      \hskip -\iis@chrulewidth%
      %
      % Small top-left horizontal rule above "Chapter"
      \rule[\iis@chpyy]{\iis@chpx}{\iis@chrulewidth}%
      \hskip -\iis@chpx%
      %
      % Prefix text and number
      {\iis@chapterprefixfont\chapapp\space}%
      {\iis@chapternumberfont\thechapter}%
      \hskip 1pt%
      %
      % Top rule and right vertical rule
      \iis@chapterrulefill{\iis@chrulewidth}%
      \rule{\iis@chrulewidth}{\iis@chpyy}%
      \par\nobreak
      %
      % Bottom rule, starting after "Chapter <number>"
      \vskip -\baselineskip
      \vskip -\iis@chpyy
      \hskip \iis@chmylen
      \iis@chapterrulefill{\iis@chrulewidth}%
      \par\nobreak
      \vskip \iis@chpyy
    }%
    \par\nobreak\vskip 20pt%
  }%
  {\usekomafont{chapter}#3\par\nobreak}%
}

% Packages for typesetting tables.
\RequirePackage{tabularx}
\RequirePackage{booktabs}


% Easily customize the three basic list environments: enumerate, itemize, and description.
\RequirePackage{enumitem}

% Extra symbols through the Text Companion fonts
\RequirePackage{textcomp}

% Easier mathematical operators
% \RequirePackage{commath}

% Theorems, lemmas, corollaries, proofs, and definitions
\RequirePackage{amsthm}
%
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[chapter]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{corolary}[theorem]{Corollary}
%
\theoremstyle{definition}
\newtheorem{definition}{Definition}[chapter]

% The `IEEEeqnarray` environment allows to properly typeset complex multi-line equations.
\RequirePackage[retainorgcmds]{IEEEtrantools}
% Fix an incompatibility with `cleveref`.  Source: https://tex.stackexchange.com/a/132848/92384.
\let\if@IEEEissubequation\iffalse

% Properly typeset quantities with SI units.
\RequirePackage{siunitx}
\sisetup{
  display-per-mode = fraction,
  inline-per-mode  = symbol
}

% Enable micro-typographic enhancements such as character protrusion and font expansion.
\RequirePackage{microtype}

% `pdfpages` allows to include multi-page PDFs.
\RequirePackage{pdfpages}

% `multicol` defines the `multicols` environment, which typesets text in multiple columns.
\RequirePackage{multicol}

% `etoolbox` provides some programming tools used in this class.
\RequirePackage{etoolbox}

% `algorithm2e` provides the `algorithm` environment for typesetting algorithms.
\RequirePackage[ruled, linesnumbered]{algorithm2e}

% `listings` provides environments and commands for typesetting source code listings.
% `lstautogobble` enables automatic removal of white spaces due to indentation in the LaTeX source
% code.
\RequirePackage{listings}
\RequirePackage{lstautogobble}
% Let sequential `literate` assignments append to the literate list instead of replacing it.
% Source: https://tex.stackexchange.com/a/202340/92384
\def\lst@Literatekey#1\@nil@{\let\lst@ifxliterate\lst@if
    \expandafter\def\expandafter\lst@literate\expandafter{\lst@literate#1}}
% Define default listing settings.
\newcommand{\@linewrapchar}{\textbackslash}
\lstset{
 basicstyle        = \ttfamily,
 frame             = single,
 framexleftmargin  = 0.5em,
 captionpos        = t,
 breaklines        = true,
 breakatwhitespace = false,
 rulesepcolor      = \color{black},
 linewidth         = \linewidth,
 prebreak          = {\@linewrapchar},
 numbers           = none,
 numberstyle       = \ttfamily,
 xleftmargin       = 0.9em,
 xrightmargin      = 0.5em,
 literate          = {~}{\texttildelow}{1},
 % Allow line breaks after underscores.  Source: https://stackoverflow.com/a/1524421/5578775.
 literate          = {\_}{}{0\discretionary{\@linewrapchar}{\_}{\_}},
 autogobble        = true,
}

% Add support for glossaries.
% \RequirePackage[%
%  toc,              % Add the glossaries to the table of contents.
%  acronym,          % Add a list of acronyms.
%  section=chapter,  % Show glossary headers as chapters.
%  nonumberlist,     % Do not print the page numbers next to glossary entries.
% ]{glossaries}
% \makeglossaries

\RequirePackage{xparse}

% `hyperref` inserts hyperlinks into your document.  This package should always be loaded last.
% The `pdfusetitle` option causes `hyperref` to infer PDF metadata such as title and authors.
\RequirePackage[pdfusetitle]{hyperref}


% Do not put a frame around links or highlight them with colored text.
% Only URLs are highlighted by printing their text in blue.
\hypersetup{colorlinks=true, allcolors=., urlcolor=blue}
\if@printlayout
 % Display the PDF document in two columns with odd-numbered pages on the right.
 \hypersetup{pdfpagelayout=TwoColumnRight}
\else
 \hypersetup{pdfpagelayout=OneColumn}
\fi

% `cleveref` provides clever cross-referencing.  It breaks `hyperref`'s last-loading rule and must
% be loaded afterwards.
\RequirePackage[capitalize]{cleveref}

% Custom layout settings:
% - Use the normal serif font for all dispositions (e.g., chapters, sections, ..).
\setkomafont{disposition}{\normalfont\bfseries}

% Define macros for setting document metadata.
\newcommand{\misspar}[1]{\textcolor{red}{\bfseries $<$#1$>$}}
%
\title{\misspar{Specify title}}
%
\author{\misspar{Specify author name}}
%
\def\@email{\misspar{Specify e-mail address}}
\newcommand{\email}[1]{\def\@email{#1}}
%
\date{\misspar{Specify date}}
%
\def\@reporttype{\misspar{Specify report type}}
\newcommand{\reporttype}[1]{\def\@reporttype{#1}}
%
\def\@semester{\misspar{Specify semester and year}}
\newcommand{\semester}[1]{\def\@semester{#1}}
%
\def\@professor{\misspar{Specify professor}}
\newcommand{\professor}[1]{\def\@professor{#1}}
%
\def\@titlelogo{}
\newcommand{\titlelogo}[1]{\def\@titlelogo{#1}}
\def\@titlelogoheight{7cm}
\newcommand{\titlelogoheight}[1]{\def\@titlelogoheight{#1}}
%
\def\@advisors{}
\newcommand{\advisor}[1]{\listadd{\@advisors}{#1}}

% Define shortcut for printing the list of acronyms.
\newcommand{\listofacronyms}{%
 \setlength{\glslistdottedwidth}{.2\linewidth}
 \printglossary[type=\acronymtype,style=listdotted,title=List of Acronyms]
}

%% Mathematical operators and symbols

% Add proper spacing after `\forall`, `\exists`, etc.
\let\@forall\forall
\renewcommand{\forall}{\@forall\;}
\let\@exists\exists
\renewcommand{\exists}{\@exists\;}
% \let\@nexists\nexists
% \renewcommand{\nexists}{\@nexists\;}

% Multi-letter variables
\newcommand{\var}[1]{\mathit{#1}}

%% Other custom commands

% `\centerfloat` centers all floats (even overly wide ones) without adding space.
% Source: https://tex.stackexchange.com/a/27099/92384
\newcommand{\centerfloat}{%
 \parindent \z@
 \leftskip \z@ \@plus 1fil \@minus \textwidth
 \rightskip\leftskip
 \parfillskip \z@skip
}

% Customize title page.
\newfontfamily\iishelv{TeX Gyre Heros}[Scale=0.95]
\newcommand{\@printadvisor}[1]{%
 {} & #1 \\%
}%
%
\renewcommand{\maketitle}{%
 \KOMAoption{twoside}{false}
 \begin{titlepage}
  %
  \thispagestyle{empty}
  %
  \begin{center}
   %
   \begin{minipage}[b]{.45\linewidth}
    \vspace{0pt}\includegraphics[width=.8\linewidth]{eth_logo.pdf}
   \end{minipage}
   \hfill
   \begin{minipage}[b]{.45\textwidth}
    \flushright{%
     \footnotesize\iishelv\selectfont%
     Institut f\"ur Integrierte Systeme\\[-.25\baselineskip]%
     Integrated Systems Laboratory%
     \vspace*{-.05\baselineskip}%
    }
   \end{minipage}\\[-.2cm]
   %
   \rule{\textwidth}{.4pt}

   \vspace{.3cm}

   {%
    \Large%
    \textsc{Department of\\Information Technology and Electrical Engineering}
   }

   \vspace{.2cm}

   \@semester

   \vfill

   {%
    \rmfamily\fontsize{28}{40}\selectfont%
    \@title\\[.75\baselineskip]%
   }

   \@reporttype

   \vfill

   \ifx\@titlelogo\@empty
    \relax
   \else
    \includegraphics[height=\@titlelogoheight]{\@titlelogo}
   \fi

   \vfill

   {%
    \Large%
    \@author%
   }\\
   \@email

   \vfill

   \@date

   \vfill

   \ifx\@advisors\@empty
    \def\@advisors{\misspar{Specify advisors}}
   \fi
   \begin{tabular}{ll}
    Advisors:
     \forlistloop{\@printadvisor}{\@advisors}
    \rule{0pt}{3ex}%
    Professor: & \@professor \\
   \end{tabular}

  \end{center}
 \end{titlepage}
 \if@printlayout
  \KOMAoption{twoside}{true}
 \fi
}


\AtBeginDocument{%
 \maketitle
 % \bibliographystyle{IEEEtranN}
}

\newcommand{\abbrparen}[1]{%
  \penalty0\hspace{0pt plus .2em}(\mbox{#1})%
}
\setlength{\emergencystretch}{2em}

% \AtEndDocument{%
%  %\nocite*{}         % Also print non-cited references.
%  \glsaddallunused{} % Also print unused glossary entries.
% }

% \lstset{
%   language=Python,
%   basicstyle=\ttfamily\small,
%   breaklines=true,
%   frame=single,
%   columns=fullflexible,
% }

\newcommand{\vc}[1]{\symbfit{#1}}
\newcommand{\code}[1]{\texttt{#1}}

\RequirePackage{multirow}
\RequirePackage{todonotes}

\newcommand{\vq}{\symbfit{q}}
\newcommand{\vR}{\symbfit{R}}
\newcommand{\vv}{\symbfit{v}}
\newcommand{\vrho}{\symbfit{\rho}}
\newcommand{\va}{\symbfit{a}}
\newcommand{\cO}{\mathcal{O}}
\newcommand{\Sig}{\Sigma}
\newcommand{\sS}{\mathbf{S}}
\newcommand{\cC}{\mathcal{C}}
\newcommand{\cW}{\mathcal{W}}
\newcommand{\dw}{\,\frac{d\omega'}{2\pi}}
\newcommand{\Gtl}[1]{G^{\gtrless,#1}}
\newcommand{\gtl}[1]{g^{\gtrless,#1}}
\newcommand{\PhiT}{\tilde{\Phi}}
\DeclareMathOperator{\FFT}{FFT}
\DeclareMathOperator{\IFFT}{IFFT}
\DeclareMathOperator{\tr}{Tr}
\newcommand{\ndof}{n_{\mathrm{dof}}}
\newcommand{\nprim}{n_{\mathrm{prim}}}
\newcommand{\nsuper}{n_{\mathrm{super}}}
\newcommand{\ntrans}{n_{\mathrm{trans}}}
\newcommand{\Ncell}{N_{\mathrm{Cell}}}
\newcommand{\sse}[1]{{\Sigma_{s}^{#1}}}
\DeclareMathOperator{\diag}{diag}
\DeclareMathOperator{\vect}{vec}
\newcommand{\qp}{\vc{q}_\perp}
\newcommand{\DRp}{\Delta\vc{R}_\perp}
\newcommand{\dd}{\mathop{}\!\mathrm{d}}
\newcommand{\Phit}{\tilde{\Phi}}

\DeclareSIUnit{\angstrom}{\textup{\AA}}
\DeclareSIUnit{\rydberg}{Ry}
\DeclareSIUnit{\bohr}{bohr}

\NewDocumentCommand{\func}{v}{\path|#1|}

\newcommand{\NB}{N_{B}}
\newcommand{\NBS}{N_{BS}}

\RequirePackage{tikz}
\RequirePackage{tikz-feynman}