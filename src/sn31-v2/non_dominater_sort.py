#Funzione di nsga net per la dominanza dei parametri da scegliere


#Caratteristiche:
#Ordinamento non dominato (Pareto): Questo metodo è progettato per trovare i cosiddetti "fronti di Pareto" o "fronti non dominati". 

# Qui si considera il concetto di dominanza:
#Una soluzione p domina una soluzione q se p è almeno altrettanto buona di q in tutti i criteri e migliore in almeno uno.
#Utilizzo: La funzione raggruppa le soluzioni in fronti basati sulla dominanza. Il primo fronte (con rank = 0) 
# è composto dalle soluzioni non dominate da nessuna altra soluzione, il secondo fronte è dominato solo dalle soluzioni 
# nel primo fronte, e così via.

#Quando utilizzarla:
#Quando stai cercando soluzioni Pareto-ottimali, cioè soluzioni che non possono essere migliorate in un criterio senza peggiorare 
# in un altro.

#Quando hai a che fare con problemi multi-obiettivo e vuoi mantenere un insieme di soluzioni diversificate.





def non_dominated_sort(population):
    fronts = [[]]
    for p in population:
        p_dominated_solutions = []
        p_domination_count = 0
        
        for q in population:
            if dominates(p, q):
                p_dominated_solutions.append(q)
            elif dominates(q, p):
                p_domination_count += 1
        
        if p_domination_count == 0:
            p.rank = 0
            fronts[0].append(p)
    
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in p_dominated_solutions:
                q_domination_count -= 1
                if q_domination_count == 0:
                    q.rank = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    
    return fronts
