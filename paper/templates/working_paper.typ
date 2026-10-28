#set document(
  title: "Do Speed Cameras Make Chicago Safer? A Geospatial Causal Inference and Machine Learning Study",
  author: ("Saketh Donepudi", "Amrutha Kakumani"),
)

#let navy = rgb("#1f3a5f")
#let charcoal = rgb("#2b2f33")
#let muted = rgb("#5d6772")
#let rule = rgb("#c9d3dd")
#let pale = rgb("#eef2f6")

#set text(font: "Georgia", size: 10.3pt, fill: charcoal, lang: "en")
#set par(justify: true, leading: 0.82em, spacing: 0.9em)
#set page(paper: "us-letter", margin: (x: 1.02in, y: 0.92in))
#show heading.where(level: 2): it => block(above: 1.3em, below: 0.5em)[
  #set text(font: "Avenir Next", size: 15pt, weight: "bold", fill: navy)
  #it
]
#show heading.where(level: 3): it => block(above: 1.0em, below: 0.3em)[
  #set text(font: "Avenir Next", size: 11.2pt, weight: "bold", fill: charcoal)
  #it
]
#set image(width: 100%)
#set figure(gap: 0.60em)
#show figure.caption: set text(size: 9.2pt, style: "italic", fill: muted)
#show table: it => {
  set text(size: 8.5pt, hyphenate: false)
  it
}
#show table.cell.where(y: 0): set text(weight: "bold", fill: navy)
#set table(inset: 4.0pt, stroke: 0.35pt + rule)
#set list(spacing: 0.55em)

// Cover
#page(margin: (x: 1.15in, y: 0.85in), header: none, footer: none)[
  #align(left)[
    #text(font: "Avenir Next", size: 8.5pt, tracking: 1.6pt, fill: muted)[RESEARCH WORKING PAPER]
    #v(0.18in)
    #line(length: 2.2in, stroke: 1.2pt + navy)
    #v(1.55in)
    #text(font: "Georgia", size: 31pt, weight: "bold", fill: navy)[
      Do Speed Cameras\
      Make Chicago Safer?
    ]
    #v(0.34in)
    #text(font: "Georgia", size: 15.5pt, fill: charcoal)[
      A Geospatial Causal Inference and Machine Learning Study
    ]
    #v(1.25in)
    #text(font: "Avenir Next", size: 12.5pt, weight: "medium", fill: charcoal)[
      Saketh Donepudi\
      Amrutha Kakumani
    ]
    #v(0.55in)
    #line(length: 3.5in, stroke: 0.5pt + rule)
    #v(0.16in)
    #text(font: "Avenir Next", size: 8.8pt, fill: muted)[
      October 2026 · Chicago, Illinois — Study Location
    ]
    #v(0.08in)
    #text(font: "Avenir Next", size: 8.8pt, fill: muted)[
      Independent research working paper · Not peer reviewed
    ]
  ]
  #place(bottom + right, dx: -0.42in, dy: -0.18in)[
    #image("../figures/cover_street_grid.png", width: 2.9in)
  ]
]

#pagebreak()

// Author information
#page(margin: (x: 1.15in, y: 1.0in), header: none, footer: none)[
  #text(font: "Avenir Next", size: 8.5pt, tracking: 1.5pt, fill: muted)[AUTHORS & PUBLICATION INFORMATION]
  #v(0.18in)
  #line(length: 2.2in, stroke: 1.0pt + navy)
  #v(1.2in)
  #text(font: "Georgia", size: 18pt, weight: "bold", fill: navy)[Saketh Donepudi]
  #v(0.05in)
  #text(font: "Avenir Next", size: 10pt, fill: muted)[Co-author]
  #v(0.55in)
  #text(font: "Georgia", size: 18pt, weight: "bold", fill: navy)[Amrutha Kakumani]
  #v(0.05in)
  #text(font: "Avenir Next", size: 10pt, fill: muted)[Co-author]
  #v(1.15in)
  #line(length: 100%, stroke: 0.4pt + rule)
  #v(0.25in)
  #text(font: "Georgia", size: 9.6pt, style: "italic", fill: charcoal)[
    “This manuscript is an independent research working paper and has not undergone formal peer review.”
  ]
  #v(0.5in)
  #text(font: "Avenir Next", size: 8.8pt, fill: muted)[
    Repository\
    #link("https://github.com/amruthakakumani/Do-Chicago-speed-cameras-reduce-crashes-")
  ]
]

#pagebreak()

// Contents
#page(margin: (x: 1.02in, y: 0.92in), header: none, footer: none)[
  #text(font: "Avenir Next", size: 8.5pt, tracking: 1.5pt, fill: muted)[CONTENTS]
  #v(0.25in)
  #outline(title: none, depth: 2)
]

#pagebreak()

// Main matter
#set page(
  header: context {
    set text(font: "Avenir Next", size: 7.8pt, fill: muted)
    align(right)[Do Speed Cameras Make Chicago Safer?  ·  Donepudi & Kakumani]
  },
  footer: context {
    set text(font: "Avenir Next", size: 8pt, fill: muted)
    align(center)[#counter(page).display("1")]
  },
)
#counter(page).update(1)
#include("body.typ")
