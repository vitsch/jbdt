%% imsart.cls  —  MINIMAL LOCAL STUB for offline compilation
%% Replace with the official version from https://www.e-publications.org/ims/support/imsart.html
%% before submission to Bayesian Analysis.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{imsart}[2026/05/20 IMS article class (local stub)]

\DeclareOption{ba}{}
\DeclareOption{ejs}{}
\DeclareOption{aos}{}
\DeclareOption*{}
\ProcessOptions\relax

\LoadClass[12pt,a4paper]{article}

\RequirePackage{amsmath,amssymb,amsthm}
\RequirePackage{geometry}
\geometry{margin=3cm, top=3.5cm, bottom=3.5cm}
\RequirePackage{setspace}
\doublespacing
\RequirePackage{fancyhdr}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[R]{\thepage}
\fancyhead[L]{\footnotesize\@runtitle}
\renewcommand{\headrulewidth}{0.4pt}
\setlength{\headheight}{15pt}

% ---- Metadata ----
\newcommand{\pubyear}[1]{\gdef\@pubyear{#1}}
\newcommand{\volume}[1]{\gdef\@volume{#1}}
\newcommand{\issue}[1]{\gdef\@issue{#1}}
\newcommand{\firstpage}[1]{\gdef\@firstpage{#1}}
\newcommand{\lastpage}[1]{\gdef\@lastpage{#1}}
\newcommand{\runauthor}[1]{\gdef\@runauthor{#1}}
\newcommand{\runtitle}[1]{\gdef\@runtitle{#1}}
\pubyear{2026}\volume{00}\issue{0}\firstpage{1}\lastpage{1}
\gdef\@runauthor{Author}\gdef\@runtitle{Short title}

% ---- Author name formatting ----
\newcommand{\fnms}[1]{#1}
\newcommand{\snm}[1]{\textsc{#1}}

% \ead[label=key]{email}  or  \ead{email}
% Stores email under key (stripping "label=" prefix) and prints it inline.
\def\ead{\@ifnextchar[{\@ims@ead@keyed}{\@ims@ead@plain}}
\def\@ims@ead@plain#1{\unskip\,(\texttt{#1})}
\def\@ims@ead@keyed[#1]#2{%
  \@ims@store@email#1=\@nil{#2}%
  \unskip\,(\texttt{#2})%
}
% Parse "label=key=\@nil" and store email under key
\def\@ims@store@email label=#1=\@nil#2{%
  \expandafter\gdef\csname @ims@email@#1\endcsname{#2}%
}

% \printead{key} or \printead{key1,key2,...} — print stored emails
\newcommand{\printead}[1]{%
  \def\@ims@ead@list{#1}%
  \def\@ims@printead@sep{}%
  \@for\@ead@key:=\@ims@ead@list\do{%
    \@ims@printead@sep
    \ifcsname @ims@email@\@ead@key\endcsname
      \small\texttt{\csname @ims@email@\@ead@key\endcsname}%
    \fi
    \def\@ims@printead@sep{,\ }%
  }%
}

\newcommand{\affiliation}[1]{\textit{#1}}

% ---- Thanks (corresponding author marker) ----
\gdef\@ims@thankstext{}
\newcommand{\thanksref}[1]{\textsuperscript{$*$}}
\newcommand{\thankstext}[2]{\gdef\@ims@thankstext{#2}}

% ---- Saved content for front matter ----
\gdef\@ims@address{}
\gdef\@ims@authorline{}

\newcommand{\address}[1]{\gdef\@ims@address{#1}}

% ---- aug environment: accumulate multiple \author calls ----
\newenvironment{aug}{%
  \gdef\@ims@authorline{}%
  \gdef\@ims@authorsep{}%
  \def\author##1{%
    \g@addto@macro\@ims@authorline{\@ims@authorsep}%
    \g@addto@macro\@ims@authorline{##1}%
    \gdef\@ims@authorsep{\enspace\textbullet\enspace}%
  }%
}{%
}

% ---- abstract environment ----
\newbox\@ims@absbox
\renewenvironment{abstract}{%
  \global\setbox\@ims@absbox\vbox\bgroup
  \noindent{\small\textbf{Abstract.}\enspace}\small\ignorespaces
}{%
  \egroup
}

% ---- keyword environment ----
\newbox\@ims@kwdbox
\newenvironment{keyword}{%
  \global\setbox\@ims@kwdbox\vbox\bgroup
  \noindent{\small\textbf{Keywords:}\enspace}\small\ignorespaces
}{%
  \egroup
}
\newcommand{\kwd}[1]{#1;\space}

% ---- frontmatter environment ----
\newenvironment{frontmatter}{%
  \begingroup
}{%
  \@ims@makefront
  \endgroup
}

\newcommand{\@ims@makefront}{%
  \thispagestyle{plain}
  \begin{center}
    {\LARGE\bfseries\@title\par}%
    \vskip1.5em
    {\large\@ims@authorline}\\[0.5em]
    {\small\@ims@address}%
    \ifx\@ims@thankstext\@empty\else
      \\[0.3em]{\small\itshape$^*$\@ims@thankstext}\fi
    \vskip1em
    \hrule
    \vskip1em
  \end{center}
  \ifvoid\@ims@absbox\else\noindent\box\@ims@absbox\fi
  \vskip0.5em
  \ifvoid\@ims@kwdbox\else\noindent\box\@ims@kwdbox\fi
  \vskip1.5em
  \hrule
  \vskip2em
}

% ---- Bibliography style ----
\AtBeginDocument{%
  \IfFileExists{ba.bst}{}{%
    \let\@ims@bstyle\bibliographystyle
    \renewcommand\bibliographystyle[1]{\@ims@bstyle{plainnat}}%
  }%
}
